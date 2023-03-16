from fontTools import configLogger
import numpy as np
import time

cl_type = 1 #sequence-based curriculum
# cl_type = 3 #graph-based curriculum (No of candidates)


# Define the oomdp description
oomdp_description = {'wall': {'height' : [10,11], 'width':[10,11]}, 'trees':{'env':[0,2], 'inv':[0,2]}, 
'rocks': {'env':[0,1], 'inv':{0,1}}, 'crafting_table' : {'env':[0,1]}, 'pogo_stick':{'inv':[0,1]}}


#Define the state minimums (config)
config_state_0 = {'trees': {'env':0}, 'rocks' : {'env': 0}, 'crafting_table' : {'env':0}}
config_state_1 = {'trees': {'env':1}, 'rocks' : {'env': 0}, 'crafting_table' : {'env':0}}
config_state_2 = {'trees': {'env':0}, 'rocks' : {'env': 1}, 'crafting_table' : {'env':0}}
config_state_3 = {'trees': {'env':1}, 'rocks' : {'env': 1}, 'crafting_table' : {'env':0}}
config_state_4 = {'trees': {'env':2}, 'rocks' : {'env': 1}, 'crafting_table' : {'env':0}}
config_state_5 = {'wall': {'height' : 11, 'width':11}, 'trees': {'env':2}, 'rocks' : {'env': 1}, 'crafting_table' : {'env':1}}

complete_config_state_0 = {'wall': {'height' : 10, 'width':10},'trees': {'env':0}, 'rocks' : {'env': 0}, 'crafting_table' : {'env':0}}

#Define the state goals
goal_state_0 = {}
goal_state_1 = {'trees': {'inv':1}, 'rocks' : {'inv': 0}, 'pogo_stick' : {'inv':0}} #Get 1 tree in inventory
goal_state_2 = {'rocks': {'inv':1}, 'trees' : {'inv': 0}, 'pogo_stick' : {'inv':0}} #Get 1 rock in inventory
goal_state_3 = {'trees': {'inv':1}, 'rocks' : {'inv': 1}, 'pogo_stick' : {'inv':0}} #Get 1 tree and 1 rock in inventory
goal_state_4 = {'trees': {'inv':2}, 'rocks' : {'inv': 1}, 'pogo_stick' : {'inv':0}} #Get 2 trees and 1 rock in inventory
goal_state_5 = {'trees': {'inv':2}, 'rocks' : {'inv': 1}, 'pogo_stick' : {'inv':1}} #Get 2 trees, 1 rock and 1 pogo stick in inventory


adjacency_matrix = {
    'state_0' : {'state_1', 'state_2', 'state_3', 'state_4', 'state_5'},
    'state_1' : {'state_3', 'state_4', 'state_5'},
    'state_2' : {'state_3', 'state_4', 'state_5'},    
    'state_3' : {'state_4', 'state_5'},    
    'state_4' : {'state_5'}    
    }

start_state = 'state_0'
goal_state = 'state_5'

paths = [] #List paths of the DFA
paths.append([config_state_0, config_state_1, config_state_3, config_state_4, config_state_5])
paths.append([config_state_0, config_state_1, config_state_3, config_state_5])
paths.append([config_state_0, config_state_1, config_state_4, config_state_5])
paths.append([config_state_0, config_state_1, config_state_5])
paths.append([config_state_0, config_state_2, config_state_3, config_state_4, config_state_5])
paths.append([config_state_0, config_state_2, config_state_3, config_state_5])
paths.append([config_state_0, config_state_2, config_state_4, config_state_5])
paths.append([config_state_0, config_state_2, config_state_5])
paths.append([config_state_0, config_state_3, config_state_4, config_state_5])
paths.append([config_state_0, config_state_3, config_state_5])
paths.append([config_state_0, config_state_4, config_state_5])
paths.append([config_state_0, config_state_5])


list_configs = []
for height in range(oomdp_description['wall']['height'][0], oomdp_description['wall']['height'][1]+1):
    for width in range(oomdp_description['wall']['width'][0], oomdp_description['wall']['width'][1]+1):
        for trees in range(oomdp_description['trees']['env'][0], oomdp_description['trees']['env'][1]+1):
            for rocks in range(oomdp_description['rocks']['env'][0], oomdp_description['rocks']['env'][1]+1):
                for crafting_table in range(oomdp_description['crafting_table']['env'][0], oomdp_description['crafting_table']['env'][1]+1):
                    list_configs.append({'wall':{'height': height, 'width': width}, 'trees':{'env': trees}, 'rocks':{'env': rocks}, 'crafting_table':{'env': crafting_table}})


def generate_configs(config_1):
    final_config = []
    for config in list_configs:
        candidate = 1
        for keys in config_1:
            if keys in config and 'env' in config_1[keys].keys():
                if config_1[keys]['env'] > config[keys]['env']:
                    candidate *= 0
                else:
                    candidate *= 1

        if candidate == 1:
            final_config.append(config)

    return final_config


def check_valid_configs(config_sets):
    for config in config_sets:
        conf_len = len(config)
        if conf_len > 1:
            if config[1]['wall']['height'] <  config[0]['wall']['height'] or config[1]['wall']['width'] <  config[0]['wall']['width'] or \
                config[1]['trees']['env'] <  config[0]['trees']['env'] or  config[1]['rocks']['env'] <  config[0]['rocks']['env'] or \
                    config[1]['crafting_table']['env'] <  config[0]['crafting_table']['env']:
                    config_sets.remove(config)
                    continue
        if conf_len > 2:
            if config[2]['wall']['height'] <  config[1]['wall']['height'] or config[2]['wall']['width'] <  config[1]['wall']['width'] or \
                config[2]['trees']['env'] <  config[1]['trees']['env'] or  config[2]['rocks']['env'] <  config[1]['rocks']['env'] or \
                    config[2]['crafting_table']['env'] <  config[1]['crafting_table']['env']:
                    config_sets.remove(config)
                    continue                    
        if conf_len > 3:
            if config[3]['wall']['height'] <  config[2]['wall']['height'] or config[3]['wall']['width'] <  config[2]['wall']['width'] or \
                config[3]['trees']['env'] <  config[2]['trees']['env'] or  config[3]['rocks']['env'] <  config[2]['rocks']['env'] or \
                    config[3]['crafting_table']['env'] <  config[2]['crafting_table']['env']:
                    config_sets.remove(config)
                    continue
        if conf_len > 4:
            if config[4]['wall']['height'] <  config[3]['wall']['height'] or config[4]['wall']['width'] <  config[3]['wall']['width'] or \
                config[4]['trees']['env'] <  config[3]['trees']['env'] or  config[4]['rocks']['env'] <  config[3]['rocks']['env'] or \
                    config[4]['crafting_table']['env'] <  config[3]['crafting_table']['env']:
                    config_sets.remove(config)
                    continue
    return config_sets

def calculate_task_similarity(config_sets, path):
    overall_task_jumps = []
    overall_goal_jumps = []
    overall_jump = []
    overall_jumps = []
    jump_config_dict = []
    for config in config_sets:   
        # print("Config:", config)
        # time.sleep(1)
        task_jumps = []
        goal_jumps = [] 
        len_conf = len(config)
        for i in range(len(config)-1):
            # print("i:",i)
            height_next = config[i+1]['wall']['height']
            height_current = config[i]['wall']['height']
            height_final = config_state_5['wall']['height']
            width_next = config[i+1]['wall']['width']
            width_current = config[i]['wall']['width']
            width_final = config_state_5['wall']['width']
            trees_next = config[i+1]['trees']['env']
            trees_current = config[i]['trees']['env']
            trees_final = config_state_5['trees']['env']
            rocks_next = config[i+1]['rocks']['env']
            rocks_current = config[i]['rocks']['env']
            rocks_final = config_state_5['rocks']['env']
            table_next = config[i+1]['crafting_table']['env']
            table_current = config[i]['crafting_table']['env']
            table_final = config_state_5['crafting_table']['env']
            task_jump = ((height_next-height_current)/4) + ((width_next-width_current)/4) + ((trees_next-trees_current)/trees_final) + ((rocks_next-rocks_current)/rocks_final) + ((table_next-table_current)/table_final) 
            task_jump = task_jump / 5
            task_jumps.append(task_jump)

            if path == paths[0] or path == paths[4]: #[0,1,3,4,5] or [0,2,3,4,5] 
                if i == 0:
                    goal_jump = 1/4
                    goal_jumps.append(goal_jump)
                if i == 1:
                    goal_jump = 1/4             
                    goal_jumps.append(goal_jump)
                if i == 2:
                    goal_jump = 1/4             
                    goal_jumps.append(goal_jump)
                if i == 3:
                    goal_jump = 1/4                    
                    goal_jumps.append(goal_jump)

            if path == paths[1] or path == paths[5]: #[0,1,3,5] [0,2,3,5]
                if i == 0:
                    goal_jump = 1/4
                    goal_jumps.append(goal_jump)
                if i == 1:
                    goal_jump = 1/4             
                    goal_jumps.append(goal_jump)
                if i == 2:
                    goal_jump = 2/4             
                    goal_jumps.append(goal_jump)

            if path == paths[2] or path == paths[6]: #[0,1,4,5] or [0,2,4,5]
                if i == 0:
                    goal_jump = 1/4
                    goal_jumps.append(goal_jump)
                if i == 1:
                    goal_jump = 2/4             
                    goal_jumps.append(goal_jump)
                if i == 2:
                    goal_jump = 1/4             
                    goal_jumps.append(goal_jump)

            if path == paths[3] or path == paths[7]: #[0,1,5] or [0,2,5]
                if i == 0:
                    goal_jump = 1/4
                    goal_jumps.append(goal_jump)
                if i == 1:
                    goal_jump = 3/4             
                    goal_jumps.append(goal_jump)

            if path == paths[8]: #[0,3,4,5]
                if i == 0:
                    goal_jump = 2/4
                    goal_jumps.append(goal_jump)
                if i == 1:
                    goal_jump = 1/4             
                    goal_jumps.append(goal_jump)
                if i == 2:
                    goal_jump = 1/4             
                    goal_jumps.append(goal_jump)    
            if path == paths[9]: #[0,3,5]
                if i == 0:
                    goal_jump = 2/4
                    goal_jumps.append(goal_jump)
                if i == 1:
                    goal_jump = 2/4             
                    goal_jumps.append(goal_jump)
            if path == paths[10]: #[0,4,5]
                if i == 0:
                    goal_jump = 3/4
                    goal_jumps.append(goal_jump)
                if i == 1:
                    goal_jump = 1/4             
                    goal_jumps.append(goal_jump)
            if path == paths[11]: #[0,5]
                if i == 0:
                    goal_jump = 1
                    goal_jumps.append(goal_jump)
        overall_jump = (np.mean(task_jumps) + np.mean(goal_jumps))/2
        jump_config_dict.append({overall_jump:config})
        overall_task_jumps.append(task_jumps)
        overall_goal_jumps.append(goal_jumps)
        overall_jumps.append(overall_jump)

    min_jump = np.argmin(overall_jumps)
    return jump_config_dict[min_jump]
        # print(task_jumps)
        # print(goal_jumps)

per_path_min_config = {}

for path in paths:
    val_configs = []
    for _, config in enumerate(path):
        if config == config_state_5:
            val_configs.append([config_state_5])
        elif config == config_state_0:
            val_configs.append([complete_config_state_0])
        else:
            val_configs.append(generate_configs(config))

    len_config = len(val_configs)
    # print(len(val_configs))
    if len_config == 2:
        total_configs = len(val_configs[0])*len(val_configs[1])
        # print("final config: ", val_configs[1])
    if len_config == 3:
        total_configs = len(val_configs[0])*len(val_configs[1])*len(val_configs[2])
    if len_config == 4:
        total_configs = len(val_configs[0])*len(val_configs[1])*len(val_configs[2])*len(val_configs[3])
    if len_config == 5:
        total_configs = len(val_configs[0])*len(val_configs[1])*len(val_configs[2])*len(val_configs[3])*len(val_configs[4])
            
    # print('total configs: ', total_configs)
    config_sets = [[] for _ in range(total_configs)]

    a, b, c, d, e = 0, 0, 0, 0, 0

    reps_0 = int(total_configs/len(val_configs[0]))
    counter = 0
    for a in val_configs[0]:
        for i in range(reps_0):
            config_sets[i + counter*reps_0].append(a)
        counter += 1

    reps_1 = int(total_configs/len(val_configs[1]))
    counter = 0
    for a in val_configs[1]:
        for i in range(reps_1):
            config_sets[i + counter*reps_1].append(a)
        counter += 1
                        
    if len_config > 2:
        reps_2 = int(total_configs/len(val_configs[2]))
        counter = 0
        for a in val_configs[2]:
            for i in range(reps_2):
                config_sets[i + counter*reps_2].append(a)                                
            counter += 1

    if len_config > 3:
        reps_3 = int(total_configs/len(val_configs[3]))
        counter = 0
        for a in val_configs[3]:
            for i in range(reps_3):
                config_sets[i + counter*reps_3].append(a)        
            counter += 1

    if len_config > 4:
        reps_4 = int(total_configs/len(val_configs[4]))
        counter = 0
        for a in val_configs[4]:
            for i in range(reps_4):
                config_sets[i + counter*reps_4].append(a)        
            counter += 1

    config_sets = check_valid_configs(config_sets)
    # print("config sets length: ", len(config_sets))

    l_config = calculate_task_similarity(config_sets, path)
    # print("config {}".format(l_config))
    # print("keys: ",list(l_config.keys())[0] )
    per_path_min_config[list(l_config.keys())[0]] = l_config[list(l_config.keys())[0]]

print("\n")
print("\n")
# print("Overall config: ", per_path_min_config)
jump_vals = []
for i in list(per_path_min_config.keys()):
    jump_vals.append(i)

for i in range(cl_type):
    min_overall_jump = min(jump_vals)
    print("Curriculum candidates: ",per_path_min_config[min_overall_jump])
    print("\n")
    jump_vals.remove(min_overall_jump)

# print("Overall lowest jump score: ",min_overall_jump )
# print("\n")
