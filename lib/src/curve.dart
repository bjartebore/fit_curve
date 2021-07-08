import 'dart:math';

class Curve {
  Curve(this.point1, this.handle1, this.handle2, this.point2);
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