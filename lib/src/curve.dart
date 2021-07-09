import 'dart:math';

class Curve {
  Curve(this.point1, this.handle1, this.handle2, this.point2);

  factory Curve.fromList(List<List<double>> list) {
    assert(list.length == 4, 'list should be no more or no less then the length of 4');
    return Curve(
      Point(list[0][0], list[0][1]),
      Point(list[1][0], list[1][1]),
      Point(list[2][0], list[2][1]),
      Point(list[3][0], list[3][1]),
    );
  }

  // first point
  Point<double> point1;
  // first controlpoint
  Point<double> handle1;
  // end controlpoint
  Point<double> handle2;
  // end points
  Point<double> point2;

  /// Multiplication operator.
  Curve operator *(double operand) => Curve(point1 * operand, handle1 * operand, handle2 * operand, point2 * operand);

  /// Division operator.
  Curve operator /(double operand) => Curve(point1 * operand, handle1 * operand, handle2 * operand, point2 * operand);

}