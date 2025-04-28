def generate_rand_matrix(rows, cols, p_ones):
    import numpy as np

    total_cells = rows * cols
    number_of_ones = int(total_cells * p_ones)
    flat = np.array([1] * number_of_ones + [0] * (total_cells - number_of_ones))
    np.random.shuffle(flat)
    return flat.reshape((rows, cols))


def get_neighbors(mat):
    from collections import defaultdict
    import itertools

    n, m = mat.shape
    neighbor_dict = defaultdict(list)

    for i in range(n):
        for j in range(m):
            for dx, dy in itertools.product([-1, 0, 1], [-1, 0, 1]):
                if dx == 0 and dy == 0:
                    continue
                ni, nj = i + dx, j + dy
                if 0 <= ni < n and 0 <= nj < m:
                    neighbor_dict[(i, j)].append(mat[ni, nj])

    return neighbor_dict


def next_gen(mat):
    mat_copy = mat.copy()
    neighbors = get_neighbors(mat)

    for (i, j), neighbor_vals in neighbors.items():
        alive_neighbors = sum(neighbor_vals)

        if mat[i, j] == 1:
            if alive_neighbors < 2 or alive_neighbors > 3:
                mat_copy[i, j] = 0
        else:
            if alive_neighbors == 3:
                mat_copy[i, j] = 1

    return mat_copy


def matrix_data_collection(
    mat, gen_num, run_num, collection_dict, alive_percent, matrix_size
):

    collection_dict["gen_dict"]["alive_cells"].append(int(mat.sum()))
    collection_dict["gen_dict"]["gen"].append(gen_num)
    collection_dict["gen_dict"]["run"].append(run_num)
    collection_dict["gen_dict"]["initial_percent_alive"].append(alive_percent)
    collection_dict["gen_dict"]["matrix_size"].append(matrix_size)

    return collection_dict


def run_game(
    mat,
    run_num,
    matrix_size,
    master_data_collection_dict,
    alive_percent,
    max_steps=1000,
):
    import numpy as np

    matrix_data_collection(
        mat, 0, run_num, master_data_collection_dict, alive_percent, matrix_size
    )

    step = 1
    steady_count = 0

    while step < max_steps:
        new_mat = next_gen(mat)
        matrix_data_collection(
            new_mat,
            step,
            run_num,
            master_data_collection_dict,
            alive_percent,
            matrix_size,
        )

        if np.array_equal(new_mat, mat):
            steady_count += 1
        else:
            steady_count = 0

        if steady_count == 2:
            master_data_collection_dict["run_dict"]["term_reason"].append("steady")
            master_data_collection_dict["run_dict"]["last_gen"].append(step)
            master_data_collection_dict["run_dict"]["run"].append(run_num)
            master_data_collection_dict["run_dict"]["initial_percent_alive"].append(
                alive_percent
            )
            master_data_collection_dict["run_dict"]["matrix_size"].append(matrix_size)
            break

        if np.all(new_mat == 0):
            master_data_collection_dict["run_dict"]["term_reason"].append("dead")
            master_data_collection_dict["run_dict"]["last_gen"].append(step)
            master_data_collection_dict["run_dict"]["run"].append(run_num)
            master_data_collection_dict["run_dict"]["initial_percent_alive"].append(
                alive_percent
            )
            master_data_collection_dict["run_dict"]["matrix_size"].append(matrix_size)
            break

        mat = new_mat
        step += 1

    if step == max_steps:
        master_data_collection_dict["run_dict"]["term_reason"].append("max")
        master_data_collection_dict["run_dict"]["last_gen"].append(step)
        master_data_collection_dict["run_dict"]["run"].append(run_num)
        master_data_collection_dict["run_dict"]["initial_percent_alive"].append(
            alive_percent
        )
        master_data_collection_dict["run_dict"]["matrix_size"].append(matrix_size)


def sample_runs(number_of_sample_per_matrix_size_and_percent_alive, max_steps):
    from collections import defaultdict
    import pandas as pd
    import numpy as np

    assert isinstance(number_of_sample_per_matrix_size_and_percent_alive, int)
    sample_result_dict = defaultdict(lambda: defaultdict(list))
    for m in range(10, 110, 10):
        for i in np.arange(0.05, 1, 0.05):
            p = round(i, 2)
            for x in range(number_of_sample_per_matrix_size_and_percent_alive):
                matrix = generate_rand_matrix(m, m, p)
                run_game(matrix, x, m * m, sample_result_dict, p, max_steps)

    sample_output_dict = {k: pd.DataFrame(v) for k, v in sample_result_dict.items()}
    sample_generation_df = sample_output_dict["gen_dict"]
    sample_run_df = sample_output_dict["run_dict"]
    sample_output_df = pd.merge(
        sample_generation_df,
        sample_run_df,
        on=["run", "initial_percent_alive", "matrix_size"],
        how="left",
    )
    sample_output_df["id"] = (
        sample_output_df[["run", "initial_percent_alive", "matrix_size"]]
        .astype(str)
        .agg("-".join, axis=1)
    )
    return sample_output_df


def final_run(dicts, max_steps):
    from collections import defaultdict
    import pandas as pd
    import numpy as np

    result_dict = defaultdict(lambda: defaultdict(list))
    for m in range(10, 110, 10):
        for i in np.arange(0.05, 1, 0.05):
            p = round(i, 2)
            for x in range(dicts[m * m, p]):
                matrix = generate_rand_matrix(m, m, p)
                run_game(matrix, x, m * m, result_dict, p, max_steps)

    output_dict = {k: pd.DataFrame(v) for k, v in result_dict.items()}
    generation_df = output_dict["gen_dict"]
    run_df = output_dict["run_dict"]
    output_df = pd.merge(
        generation_df,
        run_df,
        on=["run", "initial_percent_alive", "matrix_size"],
        how="left",
    )
    output_df["id"] = (
        output_df[["run", "initial_percent_alive", "matrix_size"]]
        .astype(str)
        .agg("-".join, axis=1)
    )
    return output_df
