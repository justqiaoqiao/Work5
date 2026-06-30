import taichi as ti

# 初始化 Taichi，如果 GPU 有驱动问题，请改为 ti.init(arch=ti.cpu)
ti.init(arch=ti.gpu)

res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))
light_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
max_bounces = ti.field(ti.i32, shape=())

# 材质常量
MAT_DIFFUSE = 0
MAT_MIRROR = 1
MAT_GLASS = 2 

# --- 数学工具函数 ---
@ti.func
def normalize(v): return v / v.norm(1e-5)

@ti.func
def reflect(I, N): return I - 2.0 * I.dot(N) * N

@ti.func
def refract(I, N, eta):
    cos_i = -N.dot(I)
    sin2_t = eta * eta * (1.0 - cos_i * cos_i)
    T = ti.Vector([0.0, 0.0, 0.0])
    if sin2_t <= 1.0: # 未发生全反射
        cos_t = ti.sqrt(1.0 - sin2_t)
        T = eta * I + (eta * cos_i - cos_t) * N
    return T

# --- 几何求交函数 ---
@ti.func
def intersect_sphere(ro, rd, center, radius):
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4.0 * c
    if delta > 0:
        t1 = (-b - ti.sqrt(delta)) / 2.0
        if t1 > 0:
            t = t1
            normal = normalize((ro + rd * t) - center)
    return t, normal

@ti.func
def intersect_plane(ro, rd, plane_y):
    t = -1.0
    normal = ti.Vector([0.0, 1.0, 0.0])
    if ti.abs(rd.y) > 1e-5:
        t1 = (plane_y - ro.y) / rd.y
        if t1 > 0: t = t1
    return t, normal

@ti.func
def scene_intersect(ro, rd):
    min_t, hit_n, hit_c, hit_mat = 1e10, ti.Vector([0.0, 0.0, 0.0]), ti.Vector([0.0, 0.0, 0.0]), MAT_DIFFUSE
    
    # 1. 玻璃球
    t, n = intersect_sphere(ro, rd, ti.Vector([-1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t: min_t, hit_n, hit_c, hit_mat = t, n, ti.Vector([0.9, 0.9, 0.9]), MAT_GLASS
    
    # 2. 镜面球
    t, n = intersect_sphere(ro, rd, ti.Vector([1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t: min_t, hit_n, hit_c, hit_mat = t, n, ti.Vector([0.9, 0.9, 0.9]), MAT_MIRROR
    
    # 3. 地板
    t, n = intersect_plane(ro, rd, -1.0)
    if 0 < t < min_t:
        min_t, hit_n, hit_mat = t, n, MAT_DIFFUSE
        p = ro + rd * t
        hit_c = ti.Vector([0.3, 0.3, 0.3]) if (ti.floor(p.x*2)%2 + ti.floor(p.z*2)%2)%2 == 0 else ti.Vector([0.8, 0.8, 0.8])
    return min_t, hit_n, hit_c, hit_mat

# --- 核心渲染循环 ---
@ti.kernel
def render():
    for i, j in pixels:
        accum_color = ti.Vector([0.0, 0.0, 0.0])
        # MSAA: 4倍超采样实现抗锯齿
        for s in range(4):
            u = (i + ti.random() - res_x / 2.0) / res_y * 2.0
            v = (j + ti.random() - res_y / 2.0) / res_y * 2.0
            ro, rd = ti.Vector([0.0, 1.0, 5.0]), normalize(ti.Vector([u, v - 0.2, -1.0]))
            
            throughput, final_color = ti.Vector([1.0, 1.0, 1.0]), ti.Vector([0.0, 0.0, 0.0])
            for bounce in range(max_bounces[None]):
                t, N, c, mat = scene_intersect(ro, rd)
                if t > 1e9: final_color += throughput * ti.Vector([0.05, 0.15, 0.2]); break
                
                p = ro + rd * t
                if mat == MAT_MIRROR:
                    ro, rd = p + N * 1e-4, reflect(rd, N)
                    throughput *= 0.8 * c
                elif mat == MAT_GLASS:
                    eta = 1.0 / 1.5 if N.dot(rd) < 0 else 1.5
                    N_eff = N if N.dot(rd) < 0 else -N
                    rd = refract(rd, N_eff, eta)
                    ro = p - N_eff * 1e-4 if rd.norm() > 0 else p + N_eff * 1e-4
                    if rd.norm() == 0: rd = reflect(rd, N_eff)
                else:
                    L = normalize(light_pos[None] - p)
                    sh_t, _, _, _ = scene_intersect(p + N * 1e-4, L)
                    if sh_t > (light_pos[None] - p).norm():
                        final_color += throughput * c * ti.max(0.0, N.dot(L))
                    break
            accum_color += final_color
        pixels[i, j] = ti.math.clamp(accum_color / 4.0, 0.0, 1.0)

def main():
    window = ti.ui.Window("Ray Tracing Plus", (res_x, res_y))
    light_pos[None] = ti.Vector([2.0, 4.0, 3.0])
    max_bounces[None] = 3
    while window.running:
        render()
        window.get_canvas().set_image(pixels)
        window.show()

if __name__ == '__main__':
    main()