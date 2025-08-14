import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
from numba import njit
import time

@njit
def segment_energy_functional(I_val, J_val, K_val, ds, youngs_modulus, moment_of_inertia, horizontal_force):
    theta1 = np.arcsin((J_val - I_val) / ds)
    theta2 = np.arcsin((K_val - J_val) / ds)

    internal_strain_energy = 0.5 * youngs_modulus * moment_of_inertia * ((theta2 - theta1) / ds) ** 2 * ds
    external_force_work = horizontal_force * np.cos(theta1) * ds

    return internal_strain_energy + external_force_work


@njit
def viterbi_algorithm(vertical_level_count, segment_count_beam, y_levels, ds,
                      youngs_modulus, moment_of_inertia, horizontal_force):
    C = np.full((vertical_level_count, vertical_level_count, segment_count_beam + 1), np.inf, dtype=np.float64)
    survivor = np.full((vertical_level_count, vertical_level_count, segment_count_beam + 1), -1, dtype=np.int32)

    index_zero = vertical_level_count // 2

    for I in range(vertical_level_count):
        for J in range(vertical_level_count):
            if I == J == index_zero:
                C[I, J, 1] = 0.0
                survivor[I, J, 1] = -1
            else:
                C[I, J, 1] = np.inf

    for m in range(2, segment_count_beam + 1):
        for J in range(vertical_level_count):
            for K in range(vertical_level_count):
                best_cost = np.inf
                best_prev_I = -1

                for I in range(vertical_level_count):
                    prev_cost = C[I, J, m - 1]
                    if prev_cost == np.inf:
                        continue

                    trans_cost = segment_energy_functional(y_levels[I], y_levels[J], y_levels[K],
                                                           ds, youngs_modulus, moment_of_inertia,
                                                           horizontal_force)
                    total_cost = prev_cost + trans_cost

                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_prev_I = I

                C[J, K, m] = best_cost
                survivor[J, K, m] = best_prev_I

    min_cost = np.inf
    best_final_J = -1
    best_final_K = -1

    for J in range(vertical_level_count):
        for K in range(vertical_level_count):
            if K == J == index_zero:
                cost = C[J, K, segment_count_beam]
                if cost < min_cost:
                    min_cost = cost
                    best_final_J = J
                    best_final_K = K

    if best_final_J == -1 or min_cost == np.inf:
        return np.empty(0, dtype=y_levels.dtype)
    else:
        reconstructed_y_indices_list = []
        current_J_trace = best_final_J
        current_K_trace = best_final_K

        reconstructed_y_indices_list.append(current_K_trace)
        reconstructed_y_indices_list.append(current_J_trace)

        for m_trace in range(segment_count_beam, 1, -1):
            prev_I_trace = survivor[current_J_trace, current_K_trace, m_trace]
            if prev_I_trace == -1:
                break
            reconstructed_y_indices_list.append(prev_I_trace)
            current_K_trace = current_J_trace
            current_J_trace = prev_I_trace

        reconstructed_y_indices_list.reverse()

        y_viterbi = np.empty(len(reconstructed_y_indices_list), dtype=y_levels.dtype)
        for i in range(len(reconstructed_y_indices_list)):
            y_viterbi[i] = y_levels[reconstructed_y_indices_list[i]]

        return y_viterbi


@njit
def x_from_y(y, ds):
    dy = np.diff(y)
    dx = np.sqrt(ds ** 2 - dy ** 2)
    x = np.cumsum(dx)
    return np.concatenate((np.array([0.0]), x))


@njit
def energy_functional(y, ds, youngs_modulus, moment_of_inertia, horizontal_force):
    theta = np.arcsin(np.diff(y) / ds)
    diffs = np.diff(theta)
    v = 0.5 * youngs_modulus * moment_of_inertia * np.sum((diffs / ds) ** 2) * ds \
        + horizontal_force * np.sum(np.cos(theta)) * ds
    return v

def create_parameters_display(horizontal_force, x_displacement, segment_count_beam, vertical_level_count, length_beam,
                              youngs_modulus, moment_of_inertia, channel_width_half):
    header = f'{"PARAMETER":<25}  {"VALUE":<10}  {"UNIT":<10}'

    parameters_data = [
        ("Beam Segment Count", segment_count_beam, ""),
        ("Vertical Level Count", vertical_level_count, ""),
        ("Beam Length", length_beam, "mm"),
        ("Horizontal Force", round(horizontal_force, 2), "N"),
        ("Young's Modulus", youngs_modulus, "MPa"),
        ("Moment of Inertia", moment_of_inertia, "mm^4"),
        ("Channel Half-Width", channel_width_half, "mm"),
        ("Beam End x-Displacement", f"{x_displacement:.1f}", "mm"),
    ]

    body_lines = []
    for name, value, unit in parameters_data:
        value_str = str(value)
        body_lines.append(f"{name:<25}  {value_str:<10}  {unit:<10}")

    solver_params_string = "\n".join([header, *body_lines])

    return solver_params_string


def main(segment_count_beam, vertical_level_count, length_beam, youngs_modulus,
         thickness, channel_width, force_start, force_stop, force_step):
    force_values = np.linspace(force_start, force_stop, int((force_stop - force_start) / force_step) + 1)
    channel_width_half = channel_width / 2.0
    moment_of_inertia = round(30 * thickness ** 3 / 12, 2)

    output_folder = f"t_{thickness:.2f}__h_{channel_width}__L_{length_beam}"
    os.makedirs(output_folder, exist_ok=True)

    if vertical_level_count % 2 == 0:
        vertical_level_count += 1

    y_levels = np.linspace(-channel_width_half, channel_width_half, vertical_level_count)
    s, ds = np.linspace(0, length_beam, segment_count_beam + 1, retstep=True)

    fig, ax = plt.subplots(figsize=(14, 3))
    fig.subplots_adjust(left=0.05, right=0.98, top=0.6, bottom=0)
    line_viterbi, = ax.plot([], [], 'o--', color="grey")
    line_minimized, = ax.plot([], [], 'o--', color="blue")
    text_params = ax.text(0.0, 1.5, '', transform=ax.transAxes, fontfamily="monospace", va='bottom')

    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.grid(True)
    ax.set_aspect("equal")
    ax.set_xlim(0, length_beam)
    ax.set_ylim(-channel_width_half, channel_width_half)

    def update(frame):
        horizontal_force = float(force_values[frame])

        y_viterbi = viterbi_algorithm(vertical_level_count, segment_count_beam, y_levels, ds,
                                      youngs_modulus, moment_of_inertia, horizontal_force)

        if y_viterbi.size != (segment_count_beam + 1):
            y_viterbi = np.linspace(0.0, 0.0, segment_count_beam + 1)

        x_viterbi = x_from_y(y_viterbi, ds)

        res = minimize(
            energy_functional,
            x0=y_viterbi,
            args=(ds, youngs_modulus, moment_of_inertia, horizontal_force),
            constraints=[
                {"type": "eq", "fun": lambda y: y[-1]},
                {"type": "eq", "fun": lambda y: y[0]},
                {"type": "eq", "fun": lambda y: np.arcsin((y[1] - y[0]) / ds)},
                {"type": "eq", "fun": lambda y: np.arcsin((y[-1] - y[-2]) / ds)}
            ],
            bounds=[(-channel_width_half, channel_width_half)] * (segment_count_beam + 1),
            method="SLSQP",
            options={"ftol": 1e-9, "maxiter": 500}
        )

        y_minimized = res.x
        x_minimized = x_from_y(y_minimized, ds)

        x_displacement = length_beam - float(x_minimized[-1])
        x_minimized = x_minimized + x_displacement
        x_viterbi = x_viterbi + (length_beam - float(x_viterbi[-1]))

        line_viterbi.set_data(x_viterbi, y_viterbi)
        line_minimized.set_data(x_minimized, y_minimized)

        text_params.set_text(create_parameters_display(horizontal_force, x_displacement,
                                                       segment_count_beam, vertical_level_count, length_beam,
                                                       youngs_modulus, moment_of_inertia, channel_width_half))

        ax.legend([line_viterbi, line_minimized],
                  [f"Initial Guess    (V = {energy_functional(y_viterbi, ds, youngs_modulus, moment_of_inertia, horizontal_force):.3f} mJ)",
                   f"Minimized Energy (V = {res.fun:.3f} mJ)"],
                  loc="lower right", bbox_to_anchor=(1, 1.1), prop={"family": "monospace"})

        fig.canvas.draw()

        fname_stem = f"F_{horizontal_force:.2f}__u_{x_displacement:.1f}"
        save_path = os.path.join(output_folder, fname_stem + ".png")
        plt.savefig(save_path, dpi=200)

        return [line_viterbi, line_minimized, text_params]

    for frame in range(len(force_values)):
        update(frame)

    plt.close(fig)


def run_simulation(config):
    main(**config)


if __name__ == "__main__":
    start_time = time.time()

    configs = []
    chunk_size = 4

    for width, Fmax in ((20, 10), (16, 15), (11, 15)):
        force_start = 0.0
        steps = 8
        force_ranges = np.linspace(force_start, Fmax, steps + 1)

        for i in range(0, steps, chunk_size):
            start_f = float(force_ranges[i])
            stop_f = float(force_ranges[min(i + chunk_size, steps)])
            configs.append({
                "segment_count_beam": 50,
                "vertical_level_count": 31,
                "length_beam": 400.0,
                "youngs_modulus": 2.1e3,
                "thickness": 0.65,
                "channel_width": float(width),
                "force_start": start_f,
                "force_stop": stop_f,
                "force_step": 0.01
            })

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(run_simulation, configs)

    elapsed_min = (time.time() - start_time) / 60.0
    print(f"Finished in {elapsed_min:.2f} minutes")
