from numpy.core.einsumfunc import einsum_path
import gym_novel_gridworlds
import gym
import TurtleBot_v0


def set_parameters_for_model():
    global no_of_environmets, width_array, height_array, no_trees_array, no_rocks_array, table_array, no_fires_array,\
        tents_area, starting_trees_array, starting_rocks_array, starting_pogo_sticks_array, type_of_env_array, env_id

    no_of_environmets = 4

    width_array = [1.5,2,2.5,2.5]
    height_array = [1.5,2,2.5,2.5]
    no_trees_array = [1,1,2,2]
    no_rocks_array = [0,1,1,1]
    table_array = [0,0,0,1]
    starting_trees_array = [0,0,0,0]
    starting_rocks_array = [0,0,0,0]
    type_of_env_array = [1,1,1,2]

    # no_of_environmets = 2
    # width_array = [10,10]
    # height_array = [10,10]
    # no_trees_array = [2,2]
    # no_rocks_array = [1,1]
    # table_array = [1,1]
    # starting_trees_array = [0,0]
    # starting_rocks_array = [0,0]
    # type_of_env_array = [1,1]    

    env_id = "TurtleBot-v0"

def get_envs():
    set_parameters_for_model()
    envs = []
    for i_env, (width, height, no_tree, no_rock, no_table, start_tree, start_rock, type_of_env) in \
            enumerate(zip(width_array, height_array, no_trees_array, no_rocks_array, table_array, starting_trees_array, starting_rocks_array, type_of_env_array)):
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
