// ---------------------------------------------------------------------------
// pillarize.cpp —— 点云 pillarization 的**主机封装 + CPU 参考 + 自动对比测试**
//
// 这个文件同时是"可执行程序"，跑起来就是一个自检：
//   随机撒点 → GPU kernel 散一次 → CPU 按同样规则散一次 → 按集合比对
//
// ⚠️ 为什么必须"按集合比"而不是"按顺序比"：
//   kernel 用 atomicAdd 分配 pillar 编号和点槽位，**同一根 pillar 内点的顺序
//   取决于线程调度**，每次运行都可能不同。这是原子操作的本质，不是 bug。
//
// ⚠️ 两种情况下结果**必然不可复现**（所以测试里要避免）：
//   ① 点太多、pillar 数量超过 max_pillars → 哪些点被丢弃取决于线程顺序
//   ② 某个格子的点超过 max_pts_per_pillar → 保留哪几个点取决于线程顺序
//   本测试的参数远低于这两个上限，所以 GPU 和 CPU 应该严格一致。
//
// 用法：
//   ./build/bin/pillarize_test                 # 默认 5 万个点
//   ./build/bin/pillarize_test --points 200000 --bench
// ---------------------------------------------------------------------------
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "scatter_kernel.h"

namespace {

struct PillarResult {
    int num_pillars{0};
    std::vector<float> pillars;   // [P][max_pts][F]
    std::vector<int> coords;      // [P][4] = (batch, z, y, x)  ← spconv 约定
    std::vector<int> counts;      // [P] 该格子的**总点数**（可能 > max_pts）
};

// ---------------------------------------------------------------------------
// CPU 参考实现：完全按 kernel 的规则来，只是串行
// ---------------------------------------------------------------------------
PillarResult pillarizeCpu(const std::vector<float>& pts, int n, int stride,
                          const bev::PillarParams& p) {
    PillarResult r;
    r.pillars.assign(static_cast<size_t>(p.max_pillars) * p.max_pts_per_pillar
                         * p.num_features, 0.0f);
    r.coords.assign(static_cast<size_t>(p.max_pillars) * 4, 0);
    r.counts.assign(static_cast<size_t>(p.max_pillars), 0);

    std::map<int, int> cell_to_pid;
    int np = 0;

    for (int i = 0; i < n; ++i) {
        const float* pt = pts.data() + static_cast<size_t>(i) * stride;
        const int ix = static_cast<int>(std::floor((pt[0] - p.min_x) * p.inv_voxel_x));
        const int iy = static_cast<int>(std::floor((pt[1] - p.min_y) * p.inv_voxel_y));
        if (ix < 0 || ix >= p.grid_x || iy < 0 || iy >= p.grid_y) continue;

        const int cell = iy * p.grid_x + ix;
        int pid;
        auto it = cell_to_pid.find(cell);
        if (it == cell_to_pid.end()) {
            pid = np++;
            if (pid >= p.max_pillars) { --np; continue; }
            cell_to_pid[cell] = pid;
            r.coords[pid * 4 + 0] = 0;    // batch
            r.coords[pid * 4 + 1] = 0;    // z
            r.coords[pid * 4 + 2] = iy;   // y
            r.coords[pid * 4 + 3] = ix;   // x
        } else {
            pid = it->second;
        }

        const int slot = r.counts[pid]++;          // 和 kernel 一样：先自增再判断
        if (slot >= p.max_pts_per_pillar) continue;
        float* dst = r.pillars.data()
                     + ((static_cast<size_t>(pid) * p.max_pts_per_pillar + slot)
                        * p.num_features);
        for (int c = 0; c < p.num_features; ++c) dst[c] = pt[c];
    }
    r.num_pillars = np;
    return r;
}

// ---------------------------------------------------------------------------
// GPU 实现
// ---------------------------------------------------------------------------
PillarResult pillarizeGpu(const std::vector<float>& pts, int n, int stride,
                          const bev::PillarParams& p) {
    PillarResult r;

    const size_t cells = static_cast<size_t>(p.grid_x) * p.grid_y;
    const size_t pillar_elems = static_cast<size_t>(p.max_pillars)
                                * p.max_pts_per_pillar * p.num_features;

    float* d_pts = nullptr;
    int* d_cell = nullptr;
    float* d_pillars = nullptr;
    int* d_coords = nullptr;
    int* d_counts = nullptr;
    int* d_np = nullptr;

    cudaMalloc(&d_pts, pts.size() * sizeof(float));
    cudaMalloc(&d_cell, cells * sizeof(int));
    cudaMalloc(&d_pillars, pillar_elems * sizeof(float));
    cudaMalloc(&d_coords, static_cast<size_t>(p.max_pillars) * 4 * sizeof(int));
    cudaMalloc(&d_counts, static_cast<size_t>(p.max_pillars) * sizeof(int));
    cudaMalloc(&d_np, sizeof(int));

    cudaMemcpy(d_pts, pts.data(), pts.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_cell, 0xFF, cells * sizeof(int));    // 0xFF... = -1（kNoPillar）
    cudaMemset(d_np, 0, sizeof(int));
    cudaMemset(d_counts, 0, static_cast<size_t>(p.max_pillars) * sizeof(int));

    bev::launchScatter(d_pts, n, d_cell, d_pillars, d_coords, d_counts, d_np, p);
    cudaDeviceSynchronize();

    int np = 0;
    cudaMemcpy(&np, d_np, sizeof(int), cudaMemcpyDeviceToHost);
    if (np > p.max_pillars) np = p.max_pillars;

    r.num_pillars = np;
    r.pillars.resize(pillar_elems);
    r.coords.resize(static_cast<size_t>(p.max_pillars) * 4);
    r.counts.resize(static_cast<size_t>(p.max_pillars));
    cudaMemcpy(r.pillars.data(), d_pillars, pillar_elems * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(r.coords.data(), d_coords,
               static_cast<size_t>(p.max_pillars) * 4 * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(r.counts.data(), d_counts,
               static_cast<size_t>(p.max_pillars) * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(d_pts);
    cudaFree(d_cell);
    cudaFree(d_pillars);
    cudaFree(d_coords);
    cudaFree(d_counts);
    cudaFree(d_np);
    return r;
}

// ---------------------------------------------------------------------------
// 把结果转成"与顺序无关"的规范形式：map<cell, 排序后的点列>
// ---------------------------------------------------------------------------
using PointTuple = std::array<float, 8>;

std::map<long long, std::vector<PointTuple>> canonicalize(const PillarResult& r,
                                                          const bev::PillarParams& p) {
    std::map<long long, std::vector<PointTuple>> m;
    for (int i = 0; i < r.num_pillars; ++i) {
        const int iy = r.coords[i * 4 + 2];   // (b,z,y,x) → y
        const int ix = r.coords[i * 4 + 3];   //              x
        // 键 = 扁平化的格子编号，和 kernel 里的 `cell = iy*grid_x + ix` 完全一致。
        // 原来写 iy * 1000000 + ix（魔法数，隐含假设 ix < 1e6）：本测试不可能撞键，
        // 但同一个概念在两处用不同公式是隐患，统一成 p.grid_x。
        const long long key = static_cast<long long>(iy) * p.grid_x + ix;
        std::vector<PointTuple> list;
        const int stored = std::min(r.counts[i], p.max_pts_per_pillar);
        for (int s = 0; s < stored; ++s) {
            const float* src = r.pillars.data()
                               + ((static_cast<size_t>(i) * p.max_pts_per_pillar + s)
                                  * p.num_features);
            PointTuple t{};
            for (int c = 0; c < p.num_features; ++c) t[c] = src[c];
            list.push_back(t);
        }
        std::sort(list.begin(), list.end());
        m[key] = list;
    }
    return m;
}

bool compare(const PillarResult& a, const PillarResult& b,
             const bev::PillarParams& p, const char* what) {
    std::printf("\n[对比] GPU vs CPU（%s）\n", what);
    std::printf("  pillar 数: GPU %d  vs  CPU %d\n", a.num_pillars, b.num_pillars);
    if (a.num_pillars != b.num_pillars) {
        std::printf("  ❌ pillar 数量不一致\n");
        return false;
    }

    auto ca = canonicalize(a, p);
    auto cb = canonicalize(b, p);
    if (ca.size() != cb.size()) {
        std::printf("  ❌ 格子集合大小不一致（%zu vs %zu）\n", ca.size(), cb.size());
        return false;
    }

    size_t bad_cells = 0, bad_pts = 0, total_pts = 0;
    for (const auto& kv : ca) {
        auto it = cb.find(kv.first);
        if (it == cb.end()) { ++bad_cells; continue; }
        const auto& la = kv.second;
        const auto& lb = it->second;
        total_pts += la.size();
        if (la.size() != lb.size()) { ++bad_pts; continue; }
        for (size_t i = 0; i < la.size(); ++i) {
            if (la[i] != lb[i]) { ++bad_pts; break; }
        }
    }

    std::printf("  总点数: %zu\n", total_pts);
    if (bad_cells == 0 && bad_pts == 0) {
        std::printf("  ✅ 完全一致（按集合比较，忽略 pillar 内顺序）\n");
        return true;
    }
    std::printf("  ❌ 有 %zu 个格子缺失、%zu 个格子的点内容不同\n", bad_cells, bad_pts);
    return false;
}

/// 手算一个最简单的情形，避免"两个实现一起错"
bool sanityCheck(const bev::PillarParams& p) {
    std::printf("\n[手算校验] 3 个点，其中 2 个落在同一格\n");
    // 已知输入：voxel 0.5，min_x/min_y = -50
    // (0,0) → ix = (0+50)/0.5 = 100, iy = 100
    // (0.1,0.1) → 同样落在 (100,100)
    // (-49.9,-49.9) → ix=0.2→0, iy=0
    const int stride = 4;
    std::vector<float> pts{
        0.0f,   0.0f,   1.0f, 0.1f,
        0.1f,   0.1f,   1.2f, 0.2f,
        -49.9f, -49.9f, 1.4f, 0.3f,
    };
    const int n = 3;
    PillarResult r = pillarizeCpu(pts, n, stride, p);
    if (r.num_pillars != 2) {
        std::printf("  ❌ 期望 2 根 pillar，得到 %d\n", r.num_pillars);
        return false;
    }
    // 第一根应该是 (iy=100, ix=100) 且装了 2 个点
    bool found_a = false, found_b = false;
    for (int i = 0; i < r.num_pillars; ++i) {
        const int iy = r.coords[i * 4 + 2], ix = r.coords[i * 4 + 3];
        if (iy == 100 && ix == 100) found_a = (r.counts[i] == 2);
        if (iy == 0 && ix == 0) found_b = (r.counts[i] == 1);
    }
    if (found_a && found_b) {
        std::printf("  ✅ 格子 (100,100) 收 2 点，格子 (0,0) 收 1 点 —— 符合手算\n");
        return true;
    }
    std::printf("  ❌ 与手算不符（(100,100)->2 点: %d，(0,0)->1 点: %d）\n",
                found_a ? 1 : 0, found_b ? 1 : 0);
    return false;
}

// ---------------------------------------------------------------------------
// 定义级不变量检查 —— **不需要任何参考实现**
//
// 这是比"和另一个实现比"更强的一层：它检查的是算法定义本身。
// 即使我的 CPU 参考实现和 GPU kernel 一起错了，这一层也能抓出来。
//
//   ① 没有两个 pillar 共用同一个 (iy, ix)
//   ② 每根 pillar 的 (iy, ix) 都在 [0, grid) 内
//   ③ **每个界内的输入点，必须能在输出里找到**（按特征值匹配）
//   ④ 界外点数 == N − 存下点数（丢点必须有解释）
//   ⑤ 每根 pillar 存下的点数 ≤ max_pts_per_pillar
// ---------------------------------------------------------------------------
struct InvariantStat {
    long long n_oob{0};      // 出界的点数
    long long n_stored{0};   // 实际存下的点数
    long long n_twice{0};    // 被存了两次的点（不该发生）
};

bool checkInvariants(const std::vector<float>& pts, int n, int stride,
                     const bev::PillarParams& p, const PillarResult& r,
                     InvariantStat* out) {
    bool ok = true;
    InvariantStat st;

    // ---- ① 格子唯一 + ② 坐标在范围内 + ⑤ 容量 ----
    std::set<long long> cells;
    for (int i = 0; i < r.num_pillars; ++i) {
        const int iy = r.coords[i * 4 + 2];   // (b,z,y,x) → y
        const int ix = r.coords[i * 4 + 3];   //              x
        if (iy < 0 || iy >= p.grid_y || ix < 0 || ix >= p.grid_x) {
            if (ok) std::printf("  ❌ 格子越界: (%d, %d)\n", iy, ix);
            ok = false;
        }
        const long long key = static_cast<long long>(iy) * p.grid_x + ix;   // 同 kernel 的 cell
        if (!cells.insert(key).second) {
            if (ok) std::printf("  ❌ 格子重复: (%d, %d)\n", iy, ix);
            ok = false;
        }
        if (r.counts[i] > 0 &&
            std::min(r.counts[i], p.max_pts_per_pillar) > p.max_pts_per_pillar) {
            ok = false;
        }
    }
    if (!ok) { if (out) *out = st; return false; }

    // ---- ③④ 每个输入点都能被解释 ----
    //
    // 把"存下来的点"建成一个多重集合（按 4 个特征的浮点位模式做键），
    // 然后逐个检查输入点：
    //   · 界外 → n_oob++
    //   · 界内但没找到 → 丢点了（只有一种合法原因：超出容量，测试里已排除）
    struct Key {
        uint32_t a, b, c, d;
        bool operator<(const Key& o) const {
            if (a != o.a) return a < o.a;
            if (b != o.b) return b < o.b;
            if (c != o.c) return c < o.c;
            return d < o.d;
        }
        bool operator==(const Key& o) const {
            return a == o.a && b == o.b && c == o.c && d == o.d;
        }
    };
    auto makeKey = [](const float* v) {
        Key k;
        std::memcpy(&k.a, v + 0, 4);
        std::memcpy(&k.b, v + 1, 4);
        std::memcpy(&k.c, v + 2, 4);
        std::memcpy(&k.d, v + 3, 4);
        return k;
    };

    std::multiset<Key> stored;
    for (int i = 0; i < r.num_pillars; ++i) {
        const int stored_n = std::min(r.counts[i], p.max_pts_per_pillar);
        for (int s = 0; s < stored_n; ++s) {
            const float* src = r.pillars.data()
                               + ((static_cast<size_t>(i) * p.max_pts_per_pillar + s)
                                  * p.num_features);
            stored.insert(makeKey(src));
            ++st.n_stored;
        }
    }

    for (int i = 0; i < n; ++i) {
        const float* pt = pts.data() + static_cast<size_t>(i) * stride;
        const int ix = static_cast<int>(std::floor((pt[0] - p.min_x) * p.inv_voxel_x));
        const int iy = static_cast<int>(std::floor((pt[1] - p.min_y) * p.inv_voxel_y));
        if (ix < 0 || ix >= p.grid_x || iy < 0 || iy >= p.grid_y) {
            ++st.n_oob;
            continue;
        }
        auto it = stored.find(makeKey(pt));
        if (it == stored.end()) {
            if (ok) {
                std::printf("  ❌ 点 #%d (%.3f, %.3f) 在界内但输出里找不到 —— 被静默丢弃\n",
                            i, pt[0], pt[1]);
            }
            ok = false;
        } else {
            stored.erase(it);
        }
    }

    // ④ 界外 + 存下 应该正好等于 N
    st.n_twice = static_cast<long long>(stored.size());
    if (st.n_oob + st.n_stored != n) {
        if (ok) {
            std::printf("  ❌ 数量不守恒：界外 %lld + 存下 %lld = %lld，但输入 %d\n",
                        st.n_oob, st.n_stored, st.n_oob + st.n_stored, n);
        }
        ok = false;
    }
    if (st.n_twice != 0) {
        if (ok) std::printf("  ❌ 有 %lld 个点被存了两次\n", st.n_twice);
        ok = false;
    }

    if (out) *out = st;
    return ok;
}

}  // namespace

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    int n_points = 50000;
    bool bench = false;
    std::string dump_prefix;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        if (k == "--points" && i + 1 < argc) n_points = std::atoi(argv[++i]);
        else if (k == "--dump" && i + 1 < argc) dump_prefix = argv[++i];
        else if (k == "--bench") bench = true;
        else if (k == "-h" || k == "--help") {
            std::printf("用法: %s [--points N] [--bench] [--dump 前缀]\n"
                        "  --dump p  导出 p_points.bin / p_pillars.bin，供 Python 独立核对\n",
                        argv[0]);
            return 0;
        }
    }

    // 网格：[-50,50]² / 0.5 m = 200×200 格
    bev::PillarParams p;
    p.grid_x = 200;
    p.grid_y = 200;
    p.min_x = -50.0f;
    p.min_y = -50.0f;
    p.inv_voxel_x = 1.0f / 0.5f;
    p.inv_voxel_y = 1.0f / 0.5f;
    p.max_pts_per_pillar = 32;
    p.num_features = 4;
    p.point_stride = 4;
    p.max_pillars = 60000;      // 4 万格 → 留足余量，避免"容量溢出"引入不确定性

    std::printf("=========================================================\n");
    std::printf("点云 pillarization（scatter）自检\n");
    std::printf("=========================================================\n");
    std::printf("网格      : %d × %d（voxel %.2f m，范围 [%.0f, %.0f]²）\n",
                p.grid_x, p.grid_y, 1.0f / p.inv_voxel_x, p.min_x, -p.min_x);
    std::printf("容量      : max_pillars=%d，每根最多 %d 点，每点 %d 个特征\n",
                p.max_pillars, p.max_pts_per_pillar, p.num_features);
    std::printf("点数      : %d\n", n_points);

    // 固定种子 → 可复现
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> ux(-50.0f, 50.0f);
    std::uniform_real_distribution<float> uz(-2.0f, 2.0f);
    std::uniform_real_distribution<float> ui(0.0f, 1.0f);

    std::vector<float> pts(static_cast<size_t>(n_points) * p.point_stride);
    for (int i = 0; i < n_points; ++i) {
        pts[static_cast<size_t>(i) * 4 + 0] = ux(rng);
        pts[static_cast<size_t>(i) * 4 + 1] = ux(rng);
        pts[static_cast<size_t>(i) * 4 + 2] = uz(rng);
        pts[static_cast<size_t>(i) * 4 + 3] = ui(rng);
    }

    bool ok = sanityCheck(p);

    // ==== 第 2 层：定义级不变量 —— **对两个实现分别检查** ====
    //
    // ★ 这一层是打破"用 CPU 当标准答案"循环论证的关键：
    //   不变量不依赖任何参考实现，所以可以**分别**作用于 CPU 版和 GPU 版。
    //   如果 CPU 版自己都过不了"每个界内点都能找到"，它就根本没资格当参考。
    //
    //   换句话说：CPU 版不是"被假定正确"，而是"必须先自己通过这一关"。
    {
        const PillarResult rc_inv = pillarizeCpu(pts, n_points, p.point_stride, p);
        const PillarResult rg_inv = pillarizeGpu(pts, n_points, p.point_stride, p);
        InvariantStat is_cpu, is_gpu;

        std::printf("\n[不变量] 定义级检查（不依赖任何参考实现，两个实现各跑一遍）\n");

        std::printf("  —— CPU 参考 ——\n");
        const bool ok_cpu = checkInvariants(pts, n_points, p.point_stride, p,
                                           rc_inv, &is_cpu);
        std::printf("  —— GPU kernel ——\n");
        const bool ok_gpu = checkInvariants(pts, n_points, p.point_stride, p,
                                           rg_inv, &is_gpu);

        if (ok_cpu && ok_gpu) {
            std::printf("  ✅ 两个实现都满足：格子唯一、坐标在范围内、"
                        "每个界内点都能找到、数量守恒\n");
            std::printf("     CPU：界外 %lld + 存下 %lld = %d\n",
                        is_cpu.n_oob, is_cpu.n_stored, n_points);
            std::printf("     GPU：界外 %lld + 存下 %lld = %d\n",
                        is_gpu.n_oob, is_gpu.n_stored, n_points);
        } else {
            std::printf("  ❌ CPU 通过=%d，GPU 通过=%d\n", ok_cpu ? 1 : 0, ok_gpu ? 1 : 0);
            if (!ok_cpu) {
                std::printf("     ★ CPU 参考没过不变量 → 它不能当标准答案，"
                            "后面的对比结论一律作废\n");
            }
        }
        ok = ok_cpu && ok_gpu && ok;
    }

    // ==== 第 3 层：和 CPU 参考比（自证，只在参考本身可信时才有意义）====
    const PillarResult rc = pillarizeCpu(pts, n_points, p.point_stride, p);
    const PillarResult rg = pillarizeGpu(pts, n_points, p.point_stride, p);
    ok = compare(rg, rc, p, "全部随机点") && ok;

    // 覆盖率（有多少点真的进了 pillar）
    long long stored = 0;
    for (int i = 0; i < rg.num_pillars; ++i)
        stored += std::min(rg.counts[i], p.max_pts_per_pillar);
    std::printf("\n[统计] %d 点 → %d 根 pillar，实际存下 %lld 点（%.1f%%）\n",
                n_points, rg.num_pillars, stored,
                100.0 * static_cast<double>(stored) / static_cast<double>(n_points));
    std::printf("       平均每根 %.2f 点；最密的格子有 %d 点\n",
                static_cast<double>(stored) / std::max(rg.num_pillars, 1),
                *std::max_element(rg.counts.begin(),
                                  rg.counts.begin() + rg.num_pillars));

    // ---------------- 可选：导出给 Python 独立核对 ----------------
    if (!dump_prefix.empty()) {
        auto writeI32 = [](std::ofstream& f, int32_t v) {
            f.write(reinterpret_cast<const char*>(&v), 4);
        };
        {
            std::ofstream f(dump_prefix + "_points.bin", std::ios::binary);
            writeI32(f, n_points);
            writeI32(f, p.point_stride);
            f.write(reinterpret_cast<const char*>(pts.data()),
                    static_cast<std::streamsize>(pts.size() * sizeof(float)));
        }
        {
            std::ofstream f(dump_prefix + "_pillars.bin", std::ios::binary);
            writeI32(f, rg.num_pillars);
            writeI32(f, p.max_pts_per_pillar);
            writeI32(f, p.num_features);
            for (int i = 0; i < rg.num_pillars; ++i) {
                // 导出格式固定为 (y, x, count)，与内部 (b,z,y,x) 解耦
                writeI32(f, rg.coords[i * 4 + 2]);
                writeI32(f, rg.coords[i * 4 + 3]);
                writeI32(f, rg.counts[i]);
                const int sn = std::min(rg.counts[i], p.max_pts_per_pillar);
                f.write(reinterpret_cast<const char*>(
                            rg.pillars.data()
                            + static_cast<size_t>(i) * p.max_pts_per_pillar * p.num_features),
                        static_cast<std::streamsize>(sn * p.num_features * sizeof(float)));
            }
        }
        std::printf("\n[导出] %s_points.bin / %s_pillars.bin\n"
                    "       用 scripts/verify_pillarize.py 做独立核对\n",
                    dump_prefix.c_str(), dump_prefix.c_str());
    }

    // ---------------- 性能 ----------------
    if (bench) {
        std::printf("\n[计时] GPU scatter（100 次取中位数）\n");
        const size_t cells = static_cast<size_t>(p.grid_x) * p.grid_y;
        const size_t pillar_elems = static_cast<size_t>(p.max_pillars)
                                    * p.max_pts_per_pillar * p.num_features;
        float* d_pts = nullptr;
        int* d_cell = nullptr;
        float* d_pillars = nullptr;
        int* d_coords = nullptr;
        int* d_counts = nullptr;
        int* d_np = nullptr;
        cudaMalloc(&d_pts, pts.size() * sizeof(float));
        cudaMalloc(&d_cell, cells * sizeof(int));
        cudaMalloc(&d_pillars, pillar_elems * sizeof(float));
        cudaMalloc(&d_coords, static_cast<size_t>(p.max_pillars) * 4 * sizeof(int));
        cudaMalloc(&d_counts, static_cast<size_t>(p.max_pillars) * sizeof(int));
        cudaMalloc(&d_np, sizeof(int));
        cudaMemcpy(d_pts, pts.data(), pts.size() * sizeof(float), cudaMemcpyHostToDevice);

        auto once = [&] {
            cudaMemset(d_cell, 0xFF, cells * sizeof(int));
            cudaMemset(d_np, 0, sizeof(int));
            cudaMemset(d_counts, 0, static_cast<size_t>(p.max_pillars) * sizeof(int));
            bev::launchScatter(d_pts, n_points, d_cell, d_pillars, d_coords, d_counts, d_np, p);
        };

        for (int i = 0; i < 10; ++i) once();      // 预热
        cudaDeviceSynchronize();

        std::vector<float> samples;
        for (int r = 0; r < 5; ++r) {
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            cudaEventRecord(e0);
            for (int i = 0; i < 100; ++i) once();
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, e0, e1);
            samples.push_back(ms / 100.0f);
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);
        }
        std::sort(samples.begin(), samples.end());
        std::printf("  %d 点 → %.4f ms（中位数，含每次的 memset 清表）\n",
                    n_points, samples[samples.size() / 2]);

        cudaFree(d_pts);
        cudaFree(d_cell);
        cudaFree(d_pillars);
        cudaFree(d_coords);
        cudaFree(d_counts);
        cudaFree(d_np);
    }

    std::printf("\n%s\n", ok ? "结果：✅ 通过" : "结果：❌ 失败");
    return ok ? 0 : 1;
}
