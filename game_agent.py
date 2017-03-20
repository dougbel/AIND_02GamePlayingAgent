"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import math


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def distance(x1, x2):
    dist = math.sqrt((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2)
    return dist


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
        of the given player.

        Note: this function should be called from within a Player instance as
        `self.score()` -- you should not need to call this function directly.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        player : object
            A player instance in the current game (i.e., an object corresponding to
            one of the player objects `game.__player_1__` or `game.__player_2__`.)

        Returns
        -------
        float
            The value calculated by the chosen heuristic
        """
    return multivariable_score(game, player)


def distance_score(game, player):
    """
    The main idea in this heuristic is to keep as much as possible a distance versus you opponent.
    “If you are far from it, it could be easier find a place for your next movement”
    :param game: current state of the game
    :param player: a player instance of the game
    :return: distance between player of the game
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # my heuristic is about the distance between two players

    position1 = game.get_player_location(player)
    position2 = game.get_player_location(game.get_opponent(player))

    dist = distance(position1, position2)

    return float(dist)  # This was constructed just to know how bad is this as evaluation function.


def centers_board_score(game, player):
    """
    With these Heuristic I'm trying to maintain my player near to centre’s board. We saw on videos that being at the
    centre could be a good strategy to win. The main idea here is that you can make movement in any directions (ideally)
    :param game: current state of the game
    :param player: a player instance of the game
    :return: The result of the analysis of distances between player and the centre's game. It's better tha
    my player is nearest to the centre.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    opponent = game.get_opponent(player)

    board_center = (int(game.width / 2), int(game.height / 2))

    # position of my player
    position_own = game.get_player_location(player)
    # position of opponent
    position_opp = game.get_player_location(opponent)


    # I normalize these because I get values between 0 and 100
    # if I am near from centre is better for me
    norm_distance_centre_own = 100 - distance(board_center, position_own) * 100 / math.sqrt(2 * (board_center[0]) ** 2)
    # if my opponent is far from centre is better for me
    norm_distance_centre_opp = distance(board_center, position_opp) * 100 / math.sqrt(2 * (board_center[0]) ** 2)

    # this is the score, I choose weights
    score = float(norm_distance_centre_own * .65 + norm_distance_centre_opp * .35)

    return score


def multivariable_score(game, player):
    """
    I decided construct this because this kind of problems have a lot of variables, I think that is not enough
    evaluate a position with just one variable. It is also important know that is not a good idea just analyse
    your situation on the board because your opponent and his/her situation affects you too.
    In this complex heuristic I decide also to use the distance to centre's board because being at the center could be a better way to win.
    I normalize the results of every metric in order to control that weighted average used.
    :param game: current state of the game
    :param player: a player instance of the game
    :return: The heuristic calculate a score in terms of many variable thought a weighted average
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # this heuristic tries to join two heuristic that have good results in order to improve them

    opponent = game.get_opponent(player)

    board_center = (int(game.width / 2), int(game.height / 2))  # this is the centre's board

    moves_own = game.get_legal_moves(player)  # my moves
    moves_opp = game.get_legal_moves(opponent)  # opponent's moves

    position_own = game.get_player_location(player)  # my position
    position_opp = game.get_player_location(opponent)  # opponent's position

    # I normalize these because I get values between 0 and 100

    # num of movements in terms of percentage
    norm_moves_own = len(moves_own) * 100 / 8

    # num of available moves for opponent in percentage
    norm_moves_opp = len(moves_opp) * 100 / 8

    # if I am near from centre is better for me
    norm_distance_centre_own = 100 - distance(board_center, position_own) * 100 / math.sqrt(2 * (board_center[0]) ** 2)
    # if my opponent is far from centre is better for me
    norm_distance_centre_opp = distance(board_center, position_opp) * 100 / math.sqrt(2 * (board_center[0]) ** 2)

    # this is the score, I choose weights
    score = float(
        (norm_moves_own - norm_moves_opp) * .60 + norm_distance_centre_own * .20 + norm_distance_centre_opp * .20)

    return score


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.playerInTurn = None

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        if len(legal_moves) > 0:
            best_move = legal_moves[0]  # guessing
        else:
            best_move = (-1, -1)  # no more available moves

        best_result = float("-inf")  # guessing
        depth = 0  # how deep it will go in plies

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring

            if self.iterative:

                # search by the best move on different depths
                while True:
                    depth += 1
                    result, move = getattr(self, self.method)(game, depth)


                    if result > best_result:
                        best_result = result
                        best_move = move

                    if self.time_left() < self.TIMER_THRESHOLD:
                        break

            else:  # search is not made by iterative deepening

                result, move = getattr(self, self.method)(game, self.search_depth)

                if result > best_result:
                    best_result = result
                    best_move = move


        except Timeout:
            pass  # nothing to do, worst case:  guesses

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # this version implement a recursive versions of `minimax`

        # guesses
        if maximizing_player:
            score = float("-inf")
        else:
            score = float("inf")
        move = (-1 - 1)

        # this just for know who is the player in turn
        if (self.playerInTurn == None):
            self.playerInTurn = game.active_player

        # if we are at the last ply
        if depth == 0:
            return self.score(game, self.playerInTurn), (-1, -1)

        # get al possible actions for current player
        actions = game.get_legal_moves()

        for action in actions:
            temp_score, _ = self.minimax(game.forecast_move(action), depth - 1, not maximizing_player)

            if maximizing_player:
                if temp_score > score:
                    score = temp_score  # is a better maximum score
                    move = action

            else:
                if temp_score < score:
                    score = temp_score  # is a better minimum score
                    move = action

        return score, move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # this version implement a recursive versions of `alphabeta`

        # guesses
        if maximizing_player:
            score = float("-inf")
        else:
            score = float("inf")
        move = (-1 - 1)

        # this just for know who is the player in turn
        if (self.playerInTurn == None):
            self.playerInTurn = game.active_player

        # if we are at the last ply
        if depth == 0:
            return self.score(game, self.playerInTurn), (-1, -1)

        # get al posible actions for current player
        actions = game.get_legal_moves()

        for action in actions:
            temp_score, _ = self.alphabeta(game.forecast_move(action), depth - 1, alpha, beta, not maximizing_player)

            if maximizing_player:
                if temp_score > score:
                    score = temp_score  # is a better maximum score
                    move = action

                if score >= beta:
                    break  # no more search in next branch, there's no a value to search

                alpha = max(alpha, score)

            else:
                if temp_score < score:
                    score = temp_score  # is a better minimum score
                    move = action

                if score <= alpha:
                    break  # no more search in next branch, there's no a value to search

                beta = min(beta, score)

        return score, move
