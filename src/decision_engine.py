def decide_action(predicted_state,intensity,stress,energy,time_of_day):
    # High Stress + Low Energy
    if stress>=4 and energy <=2:
        return 'rest','now'
    # High Stress
    elif stress>=4 :
        return 'box_breathing','now'
    #Low Energy
    elif energy <=2:
        return 'movement','within_15_min'
    # Focused + High Energy
    elif predicted_state == 'focused' and energy >=4:
        return 'deep_word','now'
    # Sad/Overwhelmed
    elif predicted_state in  ['overwelmed','mixed']:
        return 'journaling','tonight'
    # Calm State
    elif predicted_state == 'calm':
        return 'light_planning','later_today'
    # Default
    else:
        return 'pause','within_15_min'