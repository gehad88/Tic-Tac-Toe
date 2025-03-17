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
  List<int> board = List.filled(9, 0); // 0 - Empty, 1 - X (player), 2 - O (AI)
  String message = "Your turn!";
  bool isPlayerTurn = true;
  bool isLoading = false;
  bool isDialogShowing = false;
  final List<Color> _colors = [
    Colors.blueAccent,
    Colors.purpleAccent,
    Colors.pinkAccent
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
    setState(() {
      isLoading = true;
    });

    final url = Uri.parse('http://10.0.2.2:5000/predict');
    try {
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: json.encode({'board': board}),
      ).timeout(Duration(seconds: 5));

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        final predictedMove = data['move'];

        if (board[predictedMove] == 0 && !isBoardFull(board)) {
          setState(() {
            board[predictedMove] = 2;
            message = "Your turn!";
            isPlayerTurn = true;
            isLoading = false;
          });
          checkGameResult(); // Check game result after AI move
        }
      } else {
        setState(() {
          message = "Error: Could not get prediction.";
          isLoading = false;
        });
      }
    } catch (e) {
      setState(() {
        message = "Error: Something went wrong.";
        isLoading = false;
      });
    }
  }

  void playerMove(int index) {
    if (board[index] == 0 && isPlayerTurn && !isLoading && !isDialogShowing) {
      setState(() {
        board[index] = 1;
        isPlayerTurn = false;
        message = "Computer's turn!";
      });
      checkGameResult(); // Check game result after player move
      if (!isDialogShowing) {
        getPredictedMove();
      }
    }
  }

  bool isBoardFull(List<int> board) {
    return !board.contains(0);
  }

  bool isWinningMove(List<int> board, int player) {
    for (int i = 0; i < 3; i++) {
      if (board[i * 3] == player &&
          board[i * 3 + 1] == player &&
          board[i * 3 + 2] == player) return true;
      if (board[i] == player &&
          board[i + 3] == player &&
          board[i + 6] == player) return true;
    }
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
      isDialogShowing = false;
    });
    _controller.reset();
    _controller.repeat(reverse: true);
  }

  void _showWinDialog(String winner) {
    if (!isDialogShowing) {
      setState(() {
        isDialogShowing = true;
      });

      showGeneralDialog(
        context: context,
        barrierDismissible: false,
        transitionDuration: const Duration(milliseconds: 500),
        pageBuilder: (_, __, ___) => AlertDialog(
          backgroundColor: Colors.black.withOpacity(0.8),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(20),
            side: BorderSide(color: _colors[1], width: 2),
          ),
          title: AnimatedBuilder(
            animation: _controller,
            builder: (_, child) => ShaderMask(
              shaderCallback: (bounds) => LinearGradient(
                colors: _colors,
                stops: const [0.3, 0.5, 0.7],
              ).createShader(bounds),
              child: Text(
                '$winner Wins!',
                style: const TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                ),
              ),
            ),
          ),
          content: Text(
            'Congratulations! 🎉',
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
        ),
        transitionBuilder: (_, anim, __, child) {
          return ScaleTransition(
            scale: anim,
            child: FadeTransition(
              opacity: anim,
              child: child,
            ),
          );
        },
      );
    }
  }

  void _showDrawDialog() {
    if (!isDialogShowing) {
      setState(() {
        isDialogShowing = true;
      });

      showDialog(
        context: context,
        barrierDismissible: false,
        builder: (context) => AlertDialog(
          backgroundColor: Colors.black.withOpacity(0.8),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(20),
            side: BorderSide(color: _colors[0], width: 2),
          ),
          title: const Text(
            "It's a Draw!",
            style: TextStyle(color: Colors.white, fontSize: 24),
          ),
          content: const Text(
            'Try again! 🤝',
            style: TextStyle(color: Colors.white, fontSize: 20),
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
        ),
      );
    }
  }

  void checkGameResult() {
    if (!isDialogShowing) {
      if (isWinningMove(board, 1)) {
        _showWinDialog('Player');
      } else if (isWinningMove(board, 2)) {
        _showWinDialog('Computer');
      } else if (isBoardFull(board)) {
        _showDrawDialog();
      }
    }
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
                          valueColor:
                          AlwaysStoppedAnimation<Color>(_colors[1]),
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