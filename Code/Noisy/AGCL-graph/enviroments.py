from numpy.core.einsumfunc import einsum_path
import gym_novel_gridworlds
import gym
# import TurtleBot_v0
import numpy as np

def set_parameters_for_model():
    global no_of_environmets, width_array, height_array, no_trees_array, no_rocks_array, table_array, no_fires_array,\
        tents_area, starting_trees_array, starting_rocks_array, starting_pogo_sticks_array, type_of_env_array, env_id, width_std, height_std, trees_std, rocks_std

    # no_of_environmets = 4
    # width_array = [6,8,10,12]
    # height_array = [6,8,10,12]
    # no_trees_array = [1,1,2,4]
    # no_rocks_array = [0,1,1,2]
    # table_array = [0,0,1,1]
    # starting_trees_array = [0,0,0,0]
    # starting_rocks_array = [0,0,0,0]
    # type_of_env_array = [1,1,1,2]

    no_of_environmets = 6
    width_array = [6,6,8,10,12,12]
    height_array = [6,6,8,10,12,12]
    no_trees_array = [1,0,2,1,3,4]
    no_rocks_array = [0,1,0,1,2,2]
    table_array = [0,0,0,1,1,1]
    starting_trees_array = [0,0,0,0,0,0]
    starting_rocks_array = [0,0,0,0,0,0]
    type_of_env_array = [1,1,1,1,1,2]    
    width_std = 1
    height_std = 1
    trees_std = 4/6
    rocks_std = 2/6
    # no_of_environmets = 1
    # width_array = [10]
    # height_array = [10]
    # no_trees_array = [2]
    # no_rocks_array = [1]
    # table_array = [1]
    # starting_trees_array = [0]
    # starting_rocks_array = [0]
    # type_of_env_array = [1]    

    env_id = "NovelGridworld-v0"

def get_envs():
    set_parameters_for_model()
    envs = []
    for i_env, (width, height, no_tree, no_rock, no_table, start_tree, start_rock, type_of_env) in \
            enumerate(zip(width_array, height_array, no_trees_array, no_rocks_array, table_array, starting_trees_array, starting_rocks_array, type_of_env_array)):

        width = round(width + np.random.normal(-width_std, width_std))
        if width < 8:
            width = 8
        if width > 12:
            width = 12

        height = round(height + np.random.normal(-height_std, height_std))
        if height < 8:
            height = 8
        if height > 12:
            height = 12

        no_trees = round(no_tree + np.random.normal(-trees_std,trees_std))
        if no_trees < 0:
            no_trees = 0
        if no_trees > 4:
            no_trees = 4

        no_rocks = round(no_rock + np.random.normal(-rocks_std,rocks_std))
        if no_rocks < 0:
            no_rocks = 0
        if no_rocks > 2:
            no_rocks = 2

        if type_of_env == 0 or type_of_env == 1:
            if no_trees == 0 and no_rocks == 0:
                no_trees = 1

        if type_of_env == 2:
            if no_trees < 2:
                no_trees = 2
            if no_rocks < 1:
                no_rocks = 1     

        env = gym.make(env_id,
                       map_width=width,
                       map_height=height,
                       items_quantity={'tree': no_tree, 'rock': no_rock, 'crafting_table': no_table,
                                       'pogo_stick': 0},
                       initial_inventory={'wall': 0, 'tree': start_tree, 'rock': start_rock,
                                          'crafting_table': 0, 'pogo_stick': 0,
                                          },
                       goal_env=type_of_env,
                       is_final=False)
        envs.append(env)
    return envs


if __name__ == '__main__':
    get_envs()

    # why fire is passed in two places in gym.make?
    # why pogo stick, tent and tent area are zero. The same for crafting table
    # difference between tent and tent area
    # should initial crafting table be all 0, because of assertion statements
