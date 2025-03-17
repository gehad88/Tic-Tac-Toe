import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AI Tic Tac Toe',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.deepPurple,
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
      ),
      home: const TicTacToeGame(),
    );
  }
}

class TicTacToeGame extends StatefulWidget {
  const TicTacToeGame({super.key});

  @override
  _TicTacToeGameState createState() => _TicTacToeGameState();
}

class _TicTacToeGameState extends State<TicTacToeGame>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  List<int> board = List.filled(9, 0);
  String message = "Your turn!";
  bool isPlayerTurn = true;
  bool isLoading = false;
  bool gameEnded = false;
  final List<Color> _colors = [
    Colors.blueAccent,
    Colors.purpleAccent,
    Colors.orange
  ];

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1000),
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Future<void> getPredictedMove() async {
    if (gameEnded) return;

    setState(() {
      isLoading = true;
    });

    final url = Uri.parse('http://10.0.2.2:5000/predict');
    try {
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: json.encode({'board': board}),
      ).timeout(const Duration(seconds: 5));

      if (response.statusCode == 200 && !gameEnded) {
        final data = json.decode(response.body);
        final predictedMove = data['move'];

        if (board[predictedMove] == 0 && !isBoardFull(board)) {
          setState(() {
            board[predictedMove] = 2;
            message = "Your turn!";
            isPlayerTurn = true;
            isLoading = false;
          });

          // Check if AI won
          if (isWinningMove(board, 2)) {
            setState(() {
              gameEnded = true;
              message = "Game Over - Computer Wins!";
            });
            Future.delayed(const Duration(milliseconds: 100), () {
              if (mounted) {
                showGameOverDialog('Computer');
              }
            });
          } else if (isBoardFull(board)) {
            setState(() {
              gameEnded = true;
              message = "Game Over - It's a Draw!";
            });
            Future.delayed(const Duration(milliseconds: 100), () {
              if (mounted) {
                showGameOverDialog('Draw');
              }
            });
          }
        }
      } else {
        setState(() {
          message = "Error: Could not get prediction.";
          isLoading = false;
        });
      }
    } catch (e) {
      if (!gameEnded && mounted) {
        setState(() {
          message = "Error: Something went wrong.";
          isLoading = false;
        });
      }
    }
  }

  void playerMove(int index) {
    if (board[index] == 0 && isPlayerTurn && !isLoading && !gameEnded) {
      setState(() {
        board[index] = 1;
        isPlayerTurn = false;
        message = "Computer's turn!";
      });

      // Check if player won
      if (isWinningMove(board, 1)) {
        setState(() {
          gameEnded = true;
          message = "Game Over - You Win!";
        });
        Future.delayed(const Duration(milliseconds: 100), () {
          if (mounted) {
            showGameOverDialog('Player');
          }
        });
      } else if (isBoardFull(board)) {
        setState(() {
          gameEnded = true;
          message = "Game Over - It's a Draw!";
        });
        Future.delayed(const Duration(milliseconds: 100), () {
          if (mounted) {
            showGameOverDialog('Draw');
          }
        });
      } else {
        getPredictedMove();
      }
    }
  }

  bool isBoardFull(List<int> board) {
    return !board.contains(0);
  }

  bool isWinningMove(List<int> board, int player) {
    // Check rows
    for (int i = 0; i < 9; i += 3) {
      if (board[i] == player && board[i + 1] == player && board[i + 2] == player) {
        return true;
      }
    }
    // Check columns
    for (int i = 0; i < 3; i++) {
      if (board[i] == player && board[i + 3] == player && board[i + 6] == player) {
        return true;
      }
    }
    // Check diagonals
    if (board[0] == player && board[4] == player && board[8] == player) {
      return true;
    }
    if (board[2] == player && board[4] == player && board[6] == player) {
      return true;
    }
    return false;
  }

  void resetGame() {
    setState(() {
      board = List.filled(9, 0);
      isPlayerTurn = true;
      message = "Your turn!";
      isLoading = false;
      gameEnded = false;
    });
  }

  void showGameOverDialog(String result) {
    if (!mounted) return;

    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (BuildContext context) {
        String title;
        String content;

        switch (result) {
          case 'Player':
            title = 'Congratulations!';
            content = 'You Win! 🎉';
            break;
          case 'Computer':
            title = 'Game Over';
            content = 'Computer Wins! 🤖';
            break;
          default:
            title = "It's a Draw!";
            content = 'Good Game! 🤝';
        }

        return AlertDialog(
          backgroundColor: Colors.black.withOpacity(0.8),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(20),
            side: BorderSide(color: _colors[1], width: 2),
          ),
          title: Text(
            title,
            style: const TextStyle(color: Colors.white, fontSize: 24),
          ),
          content: Text(
            content,
            style: TextStyle(
              fontSize: 20,
              color: _colors[2],
              fontWeight: FontWeight.bold,
            ),
          ),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.of(context).pop();
                resetGame();
              },
              child: const Text(
                'Play Again',
                style: TextStyle(color: Colors.white, fontSize: 18),
              ),
            ),
          ],
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [Colors.black, Colors.deepPurple.shade900],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
        child: Column(
          children: [
            AppBar(
              title: const Text('AI Tic Tac Toe'),
              backgroundColor: Colors.transparent,
              elevation: 0,
              centerTitle: true,
              flexibleSpace: Container(
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: _colors,
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                ),
              ),
            ),
            Expanded(
              child: Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    AnimatedBuilder(
                      animation: _controller,
                      builder: (_, child) => ShaderMask(
                        shaderCallback: (bounds) => LinearGradient(
                          colors: _colors,
                          stops: const [0.3, 0.5, 0.7],
                        ).createShader(bounds),
                        child: Text(
                          message,
                          style: const TextStyle(
                            fontSize: 24,
                            fontWeight: FontWeight.bold,
                            color: Colors.white,
                          ),
                        ),
                      ),
                    ),
                    if (isLoading)
                      Padding(
                        padding: const EdgeInsets.all(8.0),
                        child: CircularProgressIndicator(
                          valueColor: AlwaysStoppedAnimation<Color>(_colors[1]),
                        ),
                      ),
                    const SizedBox(height: 20),
                    AnimatedBuilder(
                      animation: _controller,
                      builder: (context, child) {
                        return Transform.scale(
                          scale: 1 + _controller.value * 0.05,
                          child: Container(
                            padding: const EdgeInsets.all(20),
                            decoration: BoxDecoration(
                              gradient: LinearGradient(
                                colors: _colors,
                                begin: Alignment.topLeft,
                                end: Alignment.bottomRight,
                              ),
                              borderRadius: BorderRadius.circular(20),
                              boxShadow: [
                                BoxShadow(
                                  color: _colors[1].withOpacity(0.5),
                                  blurRadius: 20,
                                  spreadRadius: 2,
                                ),
                              ],
                            ),
                            child: GridView.builder(
                              shrinkWrap: true,
                              physics: const NeverScrollableScrollPhysics(),
                              gridDelegate:
                              const SliverGridDelegateWithFixedCrossAxisCount(
                                crossAxisCount: 3,
                                mainAxisSpacing: 10,
                                crossAxisSpacing: 10,
                              ),
                              itemCount: 9,
                              itemBuilder: (context, index) {
                                return _GameCell(
                                  symbol: board[index] == 0
                                      ? ''
                                      : board[index] == 1
                                      ? 'X'
                                      : 'O',
                                  color: board[index] == 1
                                      ? _colors[0]
                                      : _colors[2],
                                  onTap: () => playerMove(index),
                                );
                              },
                            ),
                          ),
                        );
                      },
                    ),
                    const SizedBox(height: 30),
                    ElevatedButton(
                      onPressed: resetGame,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.transparent,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(15),
                        ),
                        padding: const EdgeInsets.symmetric(
                            horizontal: 30, vertical: 15),
                      ),
                      child: Ink(
                        decoration: BoxDecoration(
                          gradient: LinearGradient(
                            colors: _colors,
                            begin: Alignment.centerLeft,
                            end: Alignment.centerRight,
                          ),
                          borderRadius: BorderRadius.circular(15),
                        ),
                        child: Container(
                          padding: const EdgeInsets.all(15),
                          child: const Text(
                            'Reset Game',
                            style: TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.bold,
                              color: Colors.white,
                            ),
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _GameCell extends StatelessWidget {
  final String symbol;
  final Color color;
  final VoidCallback onTap;

  const _GameCell({
    required this.symbol,
    required this.color,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeInOut,
        decoration: BoxDecoration(
          color: Colors.black.withOpacity(0.3),
          borderRadius: BorderRadius.circular(15),
          border: Border.all(
            color: color.withOpacity(0.5),
            width: 2,
          ),
        ),
        child: AnimatedSwitcher(
          duration: const Duration(milliseconds: 500),
          transitionBuilder: (child, animation) => ScaleTransition(
            scale: animation,
            child: FadeTransition(
              opacity: animation,
              child: child,
            ),
          ),
          child: symbol.isEmpty
              ? const SizedBox.shrink()
              : Text(
            symbol,
            key: ValueKey(symbol),
            style: TextStyle(
              fontSize: 40,
              fontWeight: FontWeight.bold,
              color: color,
              shadows: [
                Shadow(
                  color: color.withOpacity(0.5),
                  blurRadius: 10,
                  offset: const Offset(0, 0),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}