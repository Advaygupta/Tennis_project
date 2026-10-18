# p is the probability that player A wins a point when A is serving
# q is the probability that player B wins a point when B is serving
p = 0.5
q = 0.5

import numpy as np

# please enable this while number of matches is small, as it will log the details of each match in a file named "game_log.txt".
logging_enabled = True
log_file = open("game_log.txt", "w")

def play_point(server,p,q):
    if server == 'A':
        if np.random.rand() < p:
            return 'A'
        else:
            return 'B'
    else:
        if np.random.rand() < q:
            return 'B'
        else:
            return 'A'

def play_game(server,p,q):
    score_a = 0
    score_b = 0
    while True:
        point_winner = play_point(server,p,q)
        if point_winner == 'A':
            score_a += 1
        else:
            score_b += 1
        log_file.write(f"\t\t\tPoint winner: {point_winner}, current score: Player A {score_a} - Player B {score_b}\n") if logging_enabled else None
        if (score_a >= 4 and score_a - score_b >= 2) or (score_b >= 4 and score_b - score_a >= 2):
            return point_winner
        if score_a == 3 and score_b == 3:
            while True:
                point_winner = play_point(server,p,q)
                if point_winner == 'A':
                    score_a += 1
                else:
                    score_b += 1
                log_file.write(f"\t\t\tPoint winner: {point_winner}, current score: Player A {score_a} - Player B {score_b}\n") if logging_enabled else None
                if (score_a >= 4 and score_a - score_b >= 2) or (score_b >= 4 and score_b - score_a >= 2):
                    return point_winner



def play_set(p,q,initial_server='A'):
    game_score_a = 0
    game_score_b = 0
    while True:
        log_file.write(f"\t\tSimulating game {game_score_a + game_score_b + 1}, current score: Player A {game_score_a} - Player B {game_score_b}\n") if logging_enabled else None
        game_winner = play_game(initial_server,p,q)
        if game_winner == 'A':
            game_score_a += 1
            log_file.write(f"\t\t\tPlayer A wins the game, current score: Player A {game_score_a} - Player B {game_score_b}\n") if logging_enabled else None
        else:
            game_score_b += 1
            log_file.write(f"\t\t\tPlayer B wins the game, current score: Player A {game_score_a} - Player B {game_score_b}\n") if logging_enabled else None

        if (game_score_a >= 6 and game_score_a - game_score_b >= 2) or (game_score_b >= 6 and game_score_b - game_score_a >= 2):
            return 'A' if game_score_a > game_score_b else 'B'
        log_file.write(f"\t\t\tCurrent game score: Player A {game_score_a} - Player B {game_score_b}\n") if logging_enabled else None
        if game_score_a == 6 and game_score_b == 6:
            log_file.write(f"\t\tSet score is 6-6, entering tiebreaker\n") if logging_enabled else None
            server = initial_server
            last_server = server
            deuce_score_a = 0
            deuce_score_b = 0
            while True:
                point_winner = play_point(server,p,q)
                if point_winner == 'A':
                    deuce_score_a += 1
                else:
                    deuce_score_b += 1
                if (deuce_score_a >= 7 and deuce_score_a - deuce_score_b >= 2) or (deuce_score_b >= 7 and deuce_score_b - deuce_score_a >= 2):
                    return 'A' if deuce_score_a > deuce_score_b else 'B'
                log_file.write(f"\t\t\tPoint winner: {point_winner}, current score: Player A {deuce_score_a} - Player B {deuce_score_b}\n") if logging_enabled else None
                if server == 'A' and last_server == 'A':
                    server = 'B'
                elif server == 'B' and last_server == 'B':
                    server = 'A'
                last_server = server

        initial_server = 'B' if initial_server == 'A' else 'A'

def simulate_matches(num_matches,number_of_sets,p,q):
    if np.random.rand() < 0.5:
        initial_server = 'A'
    else:
        initial_server = 'B'
    player_a_wins = 0
    player_b_wins = 0
    
    for match_count in range(num_matches):
        log_file.write(f"Simulating match {match_count+1}/{num_matches}\n") if logging_enabled else None
        server = initial_server
        player_a_set_wins = 0
        player_b_set_wins = 0
        if match_count % 100000 == 0:
            print(f"Simulating match {match_count+1}/{num_matches}, completed: {(match_count+1)/num_matches*100:.2f}%")
        while not (player_a_set_wins >= number_of_sets/2 or player_b_set_wins >= number_of_sets/2):
            log_file.write(f"\tSimulating set {player_a_set_wins + player_b_set_wins + 1}, current score: Player A {player_a_set_wins} - Player B {player_b_set_wins}\n") if logging_enabled else None
            set_winner = play_set(p,q,server)
            if set_winner == 'A':
                player_a_set_wins += 1
                log_file.write(f"\tPlayer A wins the set, current score: Player A {player_a_set_wins} - Player B {player_b_set_wins}\n") if logging_enabled else None
            else:
                player_b_set_wins += 1
                log_file.write(f"\tPlayer B wins the set, current score: Player A {player_a_set_wins} - Player B {player_b_set_wins}\n") if logging_enabled else None   
            server = 'B' if server == 'A' else 'A'
        if player_a_set_wins > player_b_set_wins:
            player_a_wins += 1
            log_file.write(f"Player A wins the match, current score: Player A {player_a_wins} - Player B {player_b_wins}\n") if logging_enabled else None
        else:
            player_b_wins += 1
            log_file.write(f"Player B wins the match, current score: Player A {player_a_wins} - Player B {player_b_wins}\n") if logging_enabled else None

    return player_a_wins, player_b_wins

if __name__ == "__main__":
    num_matches = 2
    number_of_sets = 3
    player_a_wins, player_b_wins = simulate_matches(num_matches,number_of_sets,p,q)
    print(f"Player A won {player_a_wins} out of {num_matches} matches.")
    print(f"Player B won {player_b_wins} out of {num_matches} matches.")
    print(f"Estimated probability that player A wins the set: {player_a_wins/num_matches}")
    print(f"Estimated probability that player B wins the set: {player_b_wins/num_matches}")