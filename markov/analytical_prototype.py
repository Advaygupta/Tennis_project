# p = probability that player A wins a point when A is serving
# 1-p = probability that player B wins a point when A is serving
# q = probability that player B wins a point when B is serving
# 1-q = probability that player A wins a point when B is serving

import matplotlib.pyplot as plt
import numpy as np

def game_win_probab(x):
    return x**4 + 4*(1-x)*x**4 + 10*(1-x)**2*x**4 + 20*(1-x)**3*x**5/((1-2*x*(1-x)))

def set_win_probab(p,q):
    t = game_win_probab(p)
    r = 1-game_win_probab(q)

    # Set winning probabilities for player A when A is serving and B is serving respectively, for different game scores in the set.
    P_6_0 = t**3*r**3 #6-0
    P_6_1 = 3*(1-t)*t**3*r**3 + 3*(1-r)*t**4*r**2 # 6-1
    P_6_2 = 6*(1-t)**2*t**2*r**4 + 12*(1-t)*(1-r)*t**3*r**3 + 3*(1-r)**2*t**4*r**2 # 6-2
    P_6_3 = 4*(1-t)**3*t**2*r**4 + 24*(1-t)**2*(1-r)*t**3*r**3 + 24*(1-t)*(1-r)**2*t**4*r**2 + 4*(1-r)**3*t**5*r # 6-3
    P_6_4 = 5*(1-t)**4*t*r**5 + 40*(1-t)**3*(1-r)*t**2*r**4 + 60*(1-t)**2*(1-r)**2*t**3*r**3 + 20*(1-t)*(1-r)**3*t**4*r**2 + (1-r)**4*t**5*r # 6-4
    
    # for game score 7-5 and 7-6, we must calculate the probability of reaching game score 5-5 and 6-6 first, and then multiply it with the probability of winning the next game (which is 7-5 or 7-6 respectively) to get the final probability of winning the set with a score of 7-5 or 7-6.
    P_5_5 = ((1-t)**5*r**5 + 25*(1-t)**4*(1-r)*t*r**4 + 100*(1-t)**3*(1-r)**2*t**2*r**3 + 100*(1-t)**2*(1-r)**3*t**3*r**2 + 25*(1-t)*(1-r)**4*t**4*r + (1-r)**5*t**5)
    P_7_5 = r*t*P_5_5 

    # for the score to be 7-6, we must first calculate the probability of reaching the score 6-6, and then multiply it with the probability of winning the next game (which is 7-6) to get the final probability of winning the set with a score of 7-6.
    P_6_6 = (t*(1-r)+r*(1-t))*P_5_5 

    # in the last game, the probability of player A winning the game with a score of 7-6 can be calculated by considering all the possible ways in which player A can win the game with a score of 7-6, which are as follows:
    prob_set_6_6_result_7_0 = p**3*(1-q)**4
    prob_set_6_6_result_7_1 = 3*p**3*(1-p)*(1-q)**4 + 4*p**4*(1-q)**3*q
    prob_set_6_6_result_7_2 = 6*(1-p)**2*p**3*(1-q)**4 + 16*p**4*(1-p)*(1-q)**3*q + 6*p**5*(1-q)**2*q**2
    prob_set_6_6_result_7_3 = 10*(1-p)**3*p**2*(1-q)**5 + 40*(1-p)**2*p**3*(1-q)**4*q + 30*(1-p)*p**4*(1-q)**3*q**2 + 4*p**5*(1-q)**2*q**3
    prob_set_6_6_result_7_4 = 5*(1-p)**4*p*(1-q)**6 + 50*(1-p)**3*p**2*(1-q)**5*q + 100*(1-p)**2*p**3*(1-q)**4*q**2 + 50*(1-p)*p**4*(1-q)**3*q**3 + 5*p**5*(1-q)**2*q**4
    prob_set_6_6_result_7_5 = p*(1-p)**5*(1-q)**6 + 30*(1-p)**4*p**2*(1-q)**5*q + 150*(1-p)**3*p**3*(1-q)**4*q**2 + 200*(1-p)**2*p**4*(1-q)**3*q**3 + 75*(1-p)*p**5*(1-q)**2*q**4 + 6*p**6*(1-q)*q**5
    
    
    prob_set_6_6_result_6_6 = p**6*q**6 + 36*p**5*(1-p)*q**5*(1-q) + 225*p**4*(1-p)**2*q**4*(1-q)**2 + 400*p**3*(1-p)**3*q**3*(1-q)**3 + 225*p**2*(1-p)**4*q**2*(1-q)**4 + 36*p*(1-p)**5*q*(1-q)**5 + (1-p)**6*(1-q)**6 #13th game score 6-6
    p_set_6_6_result_post_6_6 = p*(1-q)/(1-(p*q + (1-p)*(1-q))) 
    p_set_6_6_result_7_6 = prob_set_6_6_result_6_6*p_set_6_6_result_post_6_6 


    #in return (incomplete)
    P_7_6 = P_6_6*(prob_set_6_6_result_7_0 + prob_set_6_6_result_7_1 + prob_set_6_6_result_7_2 + prob_set_6_6_result_7_3 + prob_set_6_6_result_7_4 + prob_set_6_6_result_7_5 + p_set_6_6_result_7_6) 

    score_7_6_probab = p*(1-q)/(1-(p*q + (1-p)*(1-q)))


    return (P_6_0 + P_6_1 + P_6_2 + P_6_3 + P_6_4 + P_7_5 + P_7_6)


# player_a_win_when_serving = float(input("Enter the probability that player A wins a point when A is serving: "))
# player_b_win_when_serving = float(input("Enter the probability that player B wins a point when B is serving: "))
# if(player_a_win_when_serving < 0 or player_a_win_when_serving > 1 or player_b_win_when_serving < 0 or player_b_win_when_serving > 1 or (player_a_win_when_serving == 0 and player_b_win_when_serving == 0) or (player_a_win_when_serving == 1 and player_b_win_when_serving == 1)):
#     print("Invalid input. Please enter a probability between 0 and 1.")
# else:
#     print(f"Probability that player A wins the set: {set_win_probab(player_a_win_when_serving, player_b_win_when_serving)}")

fig = plt.figure()
fig.suptitle('Probability that player A wins the set', fontsize=16)
fig.set_size_inches(10, 7)
ax = plt.axes(projection='3d')
p_values = [i/100 for i in range(1,100)]
q_values = [i/100 for i in range(1,100)]
P, Q = np.meshgrid(p_values, q_values)
Z = set_win_probab(P, Q)
ax.plot_surface(P, Q, Z, cmap='plasma')
ax.set_xlabel('Probability that player A wins a point when A is serving')
ax.set_ylabel('Probability that player B wins a point when B is serving')
ax.set_zlabel('Probability that player A wins the set')
ax.elev = 90
ax.azim = 0
surf = ax.plot_surface(P, Q, Z, cmap='plasma')
fig.colorbar(surf, ax=ax)
plt.savefig('tennis_set_win_probability.png')
plt.show()




