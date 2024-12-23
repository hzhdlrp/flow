#include <bits/stdc++.h>
#include <omp.h>
#include <ctime>
#include <cstdio>

using namespace std;

constexpr size_t N = 36, M = 84;
// constexpr size_t N = 14, M = 5;
constexpr size_t T = 1'000'000;
constexpr std::array<pair<int, int>, 4> deltas{{{1, 0}, {0, -1}, {0, 1}, {-1, 0}}};
mutex m;
// char field[N][M + 1] = {
//     "#####",
//     "#.  #",
//     "#.# #",
//     "#.# #",
//     "#.# #",
//     "#.# #",
//     "#.# #",
//     "#.# #",
//     "#...#",
//     "#####",
//     "#   #",
//     "#   #",
//     "#   #",
//     "#####",
// };

char field[N][M + 1] = {
        "####################################################################################",
        "#                                                                                  #",
        "#                                                                                  #",
        "#                                                                                  #",
        "#                                                                                  #",
        "#                                                                                  #",
        "#                                       .........                                  #",
        "#..............#            #           .........                                  #",
        "#..............#            #           .........                                  #",
        "#..............#            #           .........                                  #",
        "#..............#            #                                                      #",
        "#..............#            #                                                      #",
        "#..............#            #                                                      #",
        "#..............#            #                                                      #",
        "#..............#............#                                                      #",
        "#..............#............#                                                      #",
        "#..............#............#                                                      #",
        "#..............#............#                                                      #",
        "#..............#............#                                                      #",
        "#..............#............#                                                      #",
        "#..............#............#                                                      #",
        "#..............#............#                                                      #",
        "#..............#............################                     #                 #",
        "#...........................#....................................#                 #",
        "#...........................#....................................#                 #",
        "#...........................#....................................#                 #",
        "##################################################################                 #",
        "#                                                                                  #",
        "#                                                                                  #",
        "#                                                                                  #",
        "#                                                                                  #",
        "#                                                                                  #",
        "#                                                                                  #",
        "#                                                                                  #",
        "#                                                                                  #",
        "####################################################################################",
};

struct Fixed {
    constexpr Fixed(int v): v(v << 16) {}
    constexpr Fixed(float f): v(f * (1 << 16)) {}
    constexpr Fixed(double f): v(f * (1 << 16)) {}
    constexpr Fixed(): v(0) {}

    static constexpr Fixed from_raw(int32_t x) {
        Fixed ret;
        ret.v = x;
        return ret;
    }

    int32_t v;

    auto operator<=>(const Fixed&) const = default;
    bool operator==(const Fixed&) const = default;
};

static constexpr Fixed inf = Fixed::from_raw(std::numeric_limits<int32_t>::max());
static constexpr Fixed eps = Fixed::from_raw(deltas.size());

Fixed operator+(Fixed a, Fixed b) {
    return Fixed::from_raw(a.v + b.v);
}

Fixed operator-(Fixed a, Fixed b) {
    return Fixed::from_raw(a.v - b.v);
}

Fixed operator*(Fixed a, Fixed b) {
    return Fixed::from_raw(((int64_t) a.v * b.v) >> 16);
}

Fixed operator/(Fixed a, Fixed b) {
    return Fixed::from_raw(((int64_t) a.v << 16) / b.v);
}

Fixed &operator+=(Fixed &a, Fixed b) {
    return a = a + b;
}

Fixed &operator-=(Fixed &a, Fixed b) {
    return a = a - b;
}

Fixed &operator*=(Fixed &a, Fixed b) {
    return a = a * b;
}

Fixed &operator/=(Fixed &a, Fixed b) {
    return a = a / b;
}

Fixed operator-(Fixed x) {
    return Fixed::from_raw(-x.v);
}

Fixed abs(Fixed x) {
    if (x.v < 0) {
        x.v = -x.v;
    }
    return x;
}

ostream &operator<<(ostream &out, Fixed x) {
    return out << x.v / (double) (1 << 16);
}

Fixed rho[256];

Fixed p[N][M]{}, old_p[N][M];

struct VectorField {
    array<Fixed, deltas.size()> v[N][M];
    Fixed add(int x, int y, int dx, int dy, Fixed dv) {
        size_t i = ranges::find(deltas, pair(dx, dy)) - deltas.begin();
        assert(i < deltas.size());
        v[x][y][i]  += dv;
        return v[x][y][i];
    }

    Fixed get(int x, int y, int dx, int dy) {
        int i = 2*dx + dy;
        if (i == 1) return v[x][y][3];
        if (i == 2) return v[x][y][1];
        if (i == -1) return v[x][y][2];
        return v[x][y][0];
    }
};

VectorField velocity{}, velocity_flow{};
int last_use[N][M]{};
int UT = 0;

static int threads_number = 4;
mt19937 rnd(1337);

size_t iters = 0;

tuple<Fixed, bool, pair<int, int>> propagate_flow(int x, int y, pair<int, int> sink) {
    if (last_use[x][y] > UT) return {0, 0, {0, 0}};
    ++iters;
    Fixed ret = 0;
    omp_set_num_threads(threads_number);

    while (true) {
        map<pair<int, int>, pair<int, int>> parent{};
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                parent[{i, j}] = {-1, -1};
            }
        }
        parent[{x, y}] = {x, y};

        map<pair<int, int>, Fixed> minCapacity{};
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                minCapacity[{i, j}] = 0;
            }
        }
        minCapacity[{x, y}] = INT_MAX;

        #pragma omp parallel
        {
            queue<pair<int, int>> q;
            #pragma omp single
            q.push({x, y});

            while (!q.empty()) {
                int xx, yy;
                {
                    lock_guard<mutex> lockGuard(m);
                    xx = q.front().first;
                    yy = q.front().second;
                    q.pop();

                }

                volatile bool flag = false;
                #pragma omp parallel for shared(flag)
                for (auto [dx, dy]: deltas) {
                    if (flag) continue;
                    int nx = xx + dx, ny = yy + dy;
                    if (field[nx][ny] != '#') {
                        last_use[nx][ny] += 2;
                        if (parent[{nx, ny}].first == -1 &&
                            velocity.get(xx, yy, dx, dy) > velocity_flow.get(xx, yy, dx, dy)) {
                            parent[{nx, ny}] = {xx, yy};
                            minCapacity[{nx, ny}] = min(minCapacity[{xx, yy}], velocity.get(xx, yy, dx, dy) -
                                                                               velocity_flow.get(xx, yy, dx, dy));
                            if (nx == sink.first && ny == sink.second) {
                                flag = true;
                                continue;
                            }
                            if (flag) continue;
                            #pragma omp critical
                                q.push({nx, ny});
                        } else if (parent[{nx, ny}].first == -1 && velocity_flow.get(nx, ny, -dx, -dy) > 0) {
                            parent[{nx, ny}] = {xx, yy};
                            minCapacity[{nx, ny}] = min(minCapacity[{xx, yy}], velocity_flow.get(nx, ny, -dx, -dy));
                            if (nx == sink.first && ny == sink.second) {
                                flag = true;
                                continue;
                            }
                            if (flag) continue;
                            #pragma omp critical
                            q.push({nx, ny});
                        }
                    }
                }
            }
        };

        if (parent[sink].first == -1) return {0, 0, {0, 0}};

        auto increment = minCapacity[sink];
        pair<int, int> curr = sink;
        while (curr.first != x && curr.second != y) {
            pair<int, int> prev = parent[curr];
            if (velocity.get(prev.first, prev.second, (curr.first - prev.first), (curr.second - prev.second)) >
                velocity_flow.get(prev.first, prev.second, (curr.first - prev.first), (curr.second - prev.second))) {
                velocity_flow.add(prev.first, prev.second, (curr.first - prev.first), (curr.second - prev.second),
                                  increment);
            } else {
                velocity_flow.add(curr.first, curr.second, -(curr.first - prev.first), -(curr.second - prev.second),
                                  -increment);
            }
            curr = prev;
        }
        ret += increment;
    }
    return {ret, 1, sink};
}

Fixed random01() {
    return Fixed::from_raw((rnd() & ((1 << 16) - 1)));
}

void propagate_stop(int x, int y, bool force = false) {
    if (!force) {
        bool stop = true;
        for (auto [dx, dy] : deltas) {
            int nx = x + dx, ny = y + dy;
            if (field[nx][ny] != '#' && last_use[nx][ny] < UT - 1 && velocity.get(x, y, dx, dy) > 0) {
                stop = false;
                break;
            }
        }
        if (!stop) {
            return;
        }
    }
    last_use[x][y] = UT;
    for (auto [dx, dy] : deltas) {
        int nx = x + dx, ny = y + dy;
        if (field[nx][ny] == '#' || last_use[nx][ny] == UT || velocity.get(x, y, dx, dy) > 0) {
            continue;
        }
        propagate_stop(nx, ny);
    }
}

Fixed move_prob(int x, int y) {
    Fixed sum = 0;
    for (size_t i = 0; i < deltas.size(); ++i) {
        auto [dx, dy] = deltas[i];
        int nx = x + dx, ny = y + dy;
        if (field[nx][ny] == '#' || last_use[nx][ny] == UT) {
            continue;
        }
        auto v = velocity.get(x, y, dx, dy);
        if (v < 0) {
            continue;
        }
        sum += v;
    }
    return sum;
}

struct ParticleParams {
    char type;
    Fixed cur_p;

    void swap_with(int x, int y) {
        swap(field[x][y], type);
        swap(p[x][y], cur_p);
        for (auto &i : velocity.v[x][y]) {
            i = 0;
        }
    }
};

bool propagate_move(int x, int y, bool is_first) {
    last_use[x][y] = UT - is_first;
    bool ret = false;
    int nx = -1, ny = -1;
    do {
        std::array<Fixed, deltas.size()> tres;
        Fixed sum = 0;
        for (size_t i = 0; i < deltas.size(); ++i) {
            auto [dx, dy] = deltas[i];
            int nx = x + dx, ny = y + dy;
            if (field[nx][ny] == '#' || last_use[nx][ny] == UT) {
                tres[i] = sum;
                continue;
            }
            auto v = velocity.get(x, y, dx, dy);
            if (v < 0) {
                tres[i] = sum;
                continue;
            }
            sum += v;
            tres[i] = sum;
        }

        if (sum == 0) {
            break;
        }

        Fixed p = random01() * sum;
        size_t d = std::ranges::upper_bound(tres, p) - tres.begin();
        if (d >= 4) break;
        auto [dx, dy] = deltas[d];
        nx = x + dx;
        ny = y + dy;

        ret = (last_use[nx][ny] == UT - 1 || propagate_move(nx, ny, false));
    } while (!ret);
    last_use[x][y] = UT;
    for (size_t i = 0; i < deltas.size(); ++i) {
        auto [dx, dy] = deltas[i];
        int nx = x + dx, ny = y + dy;
        if (field[nx][ny] != '#' && last_use[nx][ny] < UT - 1 && velocity.get(x, y, dx, dy) < 0) {
            propagate_stop(nx, ny);
        }
    }
    if (ret) {
        if (!is_first) {
            ParticleParams pp{};
            pp.swap_with(x, y);
            pp.swap_with(nx, ny);
            pp.swap_with(x, y);
        }
    }
    return ret;
}

int dirs[N][M]{};

int main(int argc, char *argv[]) {

    if (argc > 1) {
        threads_number = atoi(argv[1]);
    }

    rho[' '] = 0.01;
    rho['.'] = 1000;
    Fixed g = 0.1;

    for (size_t x = 0; x < N; ++x) {
        for (size_t y = 0; y < M; ++y) {
            if (field[x][y] == '#')
                continue;
            for (auto [dx, dy] : deltas) {
                dirs[x][y] += (field[x + dx][y + dy] != '#');
            }
        }
    }

    double start = clock();
    for (size_t i = 0; i < T; ++i) {

        Fixed total_delta_p = 0;
        // Apply external forces
        for (size_t x = 0; x < N; ++x) {
            for (size_t y = 0; y < M; ++y) {
                if (field[x][y] == '#')
                    continue;
                if (field[x + 1][y] != '#')
                    velocity.add(x, y, 1, 0, g);
            }
        }

        // Apply forces from p
        memcpy(old_p, p, sizeof(p));
        for (size_t x = 0; x < N; ++x) {
            for (size_t y = 0; y < M; ++y) {
                if (field[x][y] == '#')
                    continue;
                for (auto [dx, dy] : deltas) {
                    int nx = x + dx, ny = y + dy;
                    if (field[nx][ny] != '#' && old_p[nx][ny] < old_p[x][y]) {
                        auto delta_p = old_p[x][y] - old_p[nx][ny];
                        auto force = delta_p;
                        auto contr = velocity.get(nx, ny, -dx, -dy);
                        if (contr * rho[(int) field[nx][ny]] >= force) {
                            velocity.add(nx, ny, -dx, -dy, - force / rho[(int) field[nx][ny]]);
                            continue;
                        }
                        force -= contr * rho[(int) field[nx][ny]];
                        contr = 0;
                        velocity.add(x, y, dx, dy, force / rho[(int) field[x][y]]);
                        p[x][y] -= force / dirs[x][y];
                        total_delta_p -= force / dirs[x][y];
                    }
                }
            }
        }

        // Make flow from velocities
        for (auto &t : velocity_flow.v) {
            for (auto &j : t) {
                for (auto &k : j) {
                    k =0;
                }
            }
        }
        bool prop = false;
        size_t cycle_iters = 0;
        iters = 0;
        std::set<pair<int, int>> a;
        vector<pair<int, int>> destinations {{0, 0}, {N-1, M-1}, {0, M-1}, {3*N/4-1, M/2}};
        do {
            ++cycle_iters;
            UT += 2;
            prop = 0;
            for (size_t x = 0; x < N; ++x) {
                for (size_t y = 0; y < M; ++y) {
                    if (field[x][y] != '#' && last_use[x][y] != UT) {
                            auto [t, local_prop, _] = propagate_flow(x, y, destinations[3]);
                            if (t > 0) {
                                prop = 1;
                            }
                    }
                }
            }
        } while (prop);

        // Recalculate p with kinetic energy
        for (size_t x = 0; x < N; ++x) {
            for (size_t y = 0; y < M; ++y) {
                if (field[x][y] == '#')
                    continue;
                for (auto [dx, dy] : deltas) {
                    auto old_v = velocity.get(x, y, dx, dy);
                    auto new_v = velocity_flow.get(x, y, dx, dy);
                    if (old_v > 0) {
                        velocity.get(x, y, dx, dy) = new_v;
                        auto force = (old_v - new_v) * rho[(int) field[x][y]];
                        if (field[x][y] == '.')
                            force *= 0.8;
                        if (field[x + dx][y + dy] == '#') {
                            p[x][y] += force / dirs[x][y];
                            total_delta_p += force / dirs[x][y];
                        } else {
                            p[x + dx][y + dy] += force / dirs[x + dx][y + dy];
                            total_delta_p += force / dirs[x + dx][y + dy];
                        }
                    }
                }
            }
        }

        UT += 2;
        prop = false;
        for (size_t x = 0; x < N; ++x) {
            for (size_t y = 0; y < M; ++y) {
                if (field[x][y] != '#' && last_use[x][y] != UT) {
                    if (random01() < move_prob(x, y)) {
                        prop = true;
                        propagate_move(x, y, true);
                    } else {
                        propagate_stop(x, y, true);
                    }
                }
            }
        }

        if (prop) {
            cout << "Tick " << i << ":\n";
            for (size_t x = 0; x < N; ++x) {
                for (size_t y = 0; y < M; ++y) {
                    if (a.count({x, y})) {
                        cout << 'X';
                    } else {
                        cout << field[x][y];
                    }
                }
                cout << '\n';
            }
//            printf("%.4lf\n", (clock() - start) / CLOCKS_PER_SEC);
        }
    }
}