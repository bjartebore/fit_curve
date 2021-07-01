// ignore_for_file: non_constant_identifier_names
/*
 *  @preserve Dart implementation of
 *  Algorithm for Automatically Fitting Digitized Curves
 *  by Philip J. Schneider
 *  "Graphics Gems", Academic Press, 1990
 *
 *  The MIT License (MIT)
 *
 *  https://github.com/soswow/fit-curves
 */

import 'dart:math';

import 'curve.dart';
import 'list_utils.dart';

typedef Points = List<Point<double>>;

const _zeroPoint = Point<double>(0, 0);

class _Report {
  _Report(this.bezCurve, this.maxError, this.splitPoint);
  final Curve bezCurve;
  final double maxError;
  final int splitPoint;
}


class _MaxError {
  _MaxError(this.maxDist, this.splitPoint);
  final double maxDist;
  final int splitPoint;
}


double _distance(Point<double> p0, Point<double> p1) {
  return sqrt(pow(p0.x - p1.x,2) + pow(p0.y - p1.y, 2));
}

double _norm(Point<double> p) {
  return sqrt(pow(p.x, 2) + pow(p.y, 2));
}

double _squaredNorm(Point<double> p) {
  return (pow(p.x, 2) + pow(p.y, 2)).toDouble();
}

Point<double> _add(Point<double> p0, Point<double> p1) {
  return Point(p0.x + p1.x, p0.y + p1.y);
}

Point<double> _addN(List<Point<double>> points) {
  return Point(
    points.fold(0, (s, p) => s + p.x),
    points.fold(0, (s, p) => s + p.y),
  );
}

Point<double> _sub(Point<double> p0, Point<double> p1) {
  return Point<double>(p0.x - p1.x, p0.y - p1.y);
}

Point<double> _scalarMul(Point<double> p, num a) {
  return Point<double>(p.x * a, p.y * a);
}

Point<double> _scalarDiv(Point<double> p, num a) {
  return Point<double>(p.x / a, p.y / a);
}

Point<double> _normalize(Point<double> p) {
  final len = _norm(p);
  return len == 0 ? p : _scalarDiv(p, len);
}

num _dot(Point<double> p0, Point<double> p1) {
  return p0.x * p1.x + p0.y * p1.y;
}


/*
 * Fit one or more Bezier curves to a set of points.
 */
List<Curve> fitCurve(Points points, int maxError) {
  if (points.length < 2) {
    return [];
  }

  final length = points.length;
  final leftTangent = _createTangent(points[1], points[0]);
  final rightTangent = _createTangent(points[length - 2], points[length - 1]);

  return _fitCubic(points, leftTangent, rightTangent, maxError);
}


/*
 * Fit a Bezier curve to a (sub)set of digitized points.
 * Your code should not call this function directly. Use {@link fitCurve} instead.
 */
List<Curve> _fitCubic(
  Points points,
  Point<double> leftTangent,
  Point<double> rightTangent,
  num error
) {
  // Max times to try iterating (to find an acceptable curve)
  const maxIterations = 20;

  // Use heuristic if region only has two points in it
  if (points.length == 2) {
    final dist = _distance(points[0], points[1]) / 3;
    return [
      Curve(
        points[0],
        _add(points[0], _scalarMul(leftTangent, dist)),
        _add(points[1], _scalarMul(rightTangent, dist)),
        points[1]
      )
    ];
  }

  // Parameterize points, and attempt to fit curve
  final u = _chordLengthParameterize(points);
  final report = _generateAndReport(points, u, u, leftTangent, rightTangent);

  var bezCurve = report.bezCurve;
  var maxError = report.maxError;
  var splitPoint = report.splitPoint;

  if (maxError == 0 || maxError < error) {
    return [bezCurve];
  }
  // If error not too large, try some reparameterization and iteration
  if (maxError < error * error) {
    var uPrime = u;
    var prevErr = maxError;
    var prevSplit = splitPoint;

    for (var i = 0; i < maxIterations; i++) {
      uPrime = _reparameterize(bezCurve, points, uPrime);

      final report = _generateAndReport(
        points,
        u,
        uPrime,
        leftTangent,
        rightTangent
      );

      bezCurve = report.bezCurve;
      maxError = report.maxError;
      splitPoint = report.splitPoint;

      if (maxError < error) {
        return [bezCurve];
      }
      // If the development of the fitted curve grinds to a halt,
      // we abort this attempt (and try a shorter curve):
      else if (splitPoint == prevSplit) {
        final errChange = maxError / prevErr;
        if (errChange > 0.9999 && errChange < 1.0001) {
          break;
        }
      }

      prevErr = maxError;
      prevSplit = splitPoint;
    }
  }

  // Fitting failed -- split at max error point and fit recursively
  final List<Curve> beziers = [];

  // To create a smooth transition from one curve segment to the next, we
  // calculate the line between the points directly before and after the
  // center, and use that as the tangent both to and from the center point.
  var centerVector = _sub(points[splitPoint - 1], points[splitPoint + 1]);
  // However, this won't work if they're the same point, because the line we
  // want to use as a tangent would be 0. Instead, we calculate the line from
  // that "double-point" to the center point, and use its tangent.
  if (centerVector.x == 0 && centerVector.y == 0) {
    // [x,y] -> [-y,x]: http://stackoverflow.com/a/4780141/1869660
    final point= _sub(points[splitPoint - 1], points[splitPoint]);
    centerVector = Point(-point.y, point.x);
  }
  final toCenterTangent = _normalize(centerVector);
  // To and from need to point in opposite directions:
  final fromCenterTangent = _scalarMul(toCenterTangent, -1);


  beziers.addAll(_fitCubic(points.slice(0, splitPoint + 1).toList(), leftTangent, toCenterTangent, error));
  beziers.addAll(_fitCubic(points.slice(splitPoint).toList(), fromCenterTangent, rightTangent, error));
  return beziers;
}

_Report _generateAndReport(
  Points points,
  List<double> paramsOrig,
  List<double> paramsPrime,
  Point<double> leftTangent,
  Point<double> rightTangent
){
  final bezCurve = _generateBezier(points, paramsPrime, leftTangent, rightTangent);
  // Find max deviation of points to fitted curve.
  // Here we always use the original parameters (from chordLengthParameterize()),
  // because we need to compare the current curve to the actual source polyline,
  // and not the currently iterated parameters which reparameterize() & generateBezier() use,
  // as those have probably drifted far away and may no longer be in ascending order.
  final err = computeMaxError(points, bezCurve, paramsOrig);

  return _Report(bezCurve, err.maxDist, err.splitPoint);
}

/*
 * Use least-squares method to find Bezier control points for region.
 */
Curve _generateBezier(
  Points points,
  List<double> parameters,
  Point<double> leftTangent,
  Point<double> rightTangent
) {
  final firstPoint = points[0];
  final lastPoint = points[points.length - 1];

  final Curve bezCurve = Curve(firstPoint, _zeroPoint, _zeroPoint, lastPoint);

  // Compute the A's
  final A = parameters.map((u) {
    final ux = 1 - u;
    return [
      _scalarMul(leftTangent, 3 * u * (ux * ux)),
      _scalarMul(rightTangent, 3 * ux * (u * u))
    ];
  }).toList();

  // Create the C and X matrices
  final List<List<double>> C = [
    [0.0, 0.0],
    [0.0, 0.0]
  ];
  final List<double> X = [0, 0];

  for (var i = 0; i < parameters.length; i+=1) {
    final u = parameters[i];

    final a = A[i][0];
    final b = A[i][1];

    C[0][0] += _dot(a, a);
    C[0][1] += _dot(a, b);
    C[1][0] += _dot(a, b);
    C[1][1] += _dot(b, b);

    final tmp = _sub(points[i], _bezierQ(Curve(firstPoint, firstPoint, lastPoint, lastPoint), u));
    X[0] += _dot(a, tmp);
    X[1] += _dot(b, tmp);
  }

  // Compute the determinants of C and X
  final det_C0_C1 = C[0][0] * C[1][1] - C[1][0] * C[0][1];
  final det_C0_X = C[0][0] * X[1] - C[1][0] * X[0];
  final det_X_C1 = X[0] * C[1][1] - X[1] * C[0][1];

  // Finally, derive alpha values
  final alpha_l = det_C0_C1 == 0 ? 0 : det_X_C1 / det_C0_C1;
  final alpha_r = det_C0_C1 == 0 ? 0 : det_C0_X / det_C0_C1;

  // If alpha negative, use the Wu/Barsky heuristic (see text).
  // If alpha is 0, you get coincident control points that lead to
  // divide by zero in any subsequent NewtonRaphsonRootFind() call.
  final segLength = _norm(_sub(firstPoint, lastPoint));
  final epsilon = 1.0e-6 * segLength;

  if (alpha_l < epsilon || alpha_r < epsilon) {
    // Fall back on standard (probably inaccurate) formula, and subdivide further if needed.
    bezCurve.handle1 = _add(firstPoint, _scalarMul(leftTangent, segLength / 3.0));
    bezCurve.handle2 = _add(lastPoint, _scalarMul(rightTangent, segLength / 3.0));
  } else {
    // First and last control points of the Bezier curve are
    // positioned exactly at the first and last data points
    // Control points 1 and 2 are positioned an alpha distance out
    // on the tangent vectors, left and right, respectively
    bezCurve.handle1 = _add(firstPoint, _scalarMul(leftTangent, alpha_l));
    bezCurve.handle2 = _add(lastPoint, _scalarMul(rightTangent, alpha_r));
  }

  return bezCurve;
}


// Evaluates cubic bezier at t, return point
Point<double> _bezierQ(Curve ctrlPoly, num t) {
  final tx = 1.0 - t;
  final pA = _scalarMul(ctrlPoly.point1, pow(tx,3));
  final pB = _scalarMul(ctrlPoly.handle1, 3 * pow(tx, 2) * t);
  final pC = _scalarMul(ctrlPoly.handle2, 3 * tx * pow(t, 2));
  final pD = _scalarMul(ctrlPoly.point2, pow(t, 3));
  return _addN([pA, pB, pC, pD]);
}

// Evaluates cubic bezier first derivative at t, return point
Point<double> _bezierQPrime(Curve ctrlPoly, num t) {
  final tx = 1.0 - t;
  final pA = _scalarMul(_sub(ctrlPoly.handle1, ctrlPoly.point1), 3 * pow(tx, 2));
  final pB = _scalarMul(_sub(ctrlPoly.handle2, ctrlPoly.handle1), 6 * tx * t);
  final pC = _scalarMul(_sub(ctrlPoly.point2, ctrlPoly.handle2), 3 * pow(t, 2));
  return _addN([pA, pB, pC]);
}

// Evaluates cubic bezier second derivative at t, return point
Point<double> _bezierQPrimePrime(Curve ctrlPoly, num t) {
  return _add(
    _scalarMul(_add(_sub(ctrlPoly.handle2, _scalarMul(ctrlPoly.handle1, 2)), ctrlPoly.point1), 6 * (1.0 - t)),
    _scalarMul(_add(_sub(ctrlPoly.point2, _scalarMul(ctrlPoly.handle2, 2)), ctrlPoly.handle1), 6 * t)
  );
}

/*
 * Given set of points and their parameterization, try to find a better parameterization.
 */
List<double> _reparameterize(Curve bezier, Points points, List<double> parameters) {
  return parameters.indexedMap((p, i, _) => _newtonRaphsonRootFind(bezier, points[i], p)).toList();
}

/*
 * Creates a vector of length 1 which shows the direction from B to A
 */
Point<double> _createTangent(Point<double> pointA, Point<double> pointB) {
  return _normalize(_sub(pointA, pointB));
}

/*
 * Use Newton-Raphson iteration to find better root.
 */
double _newtonRaphsonRootFind(Curve bez, Point<double> point, double u) {
  final d = _sub(_bezierQ(bez, u), point);
  final qprime = _bezierQPrime(bez, u);
  final numerator = _dot(d, qprime);
  final denominator = _squaredNorm(qprime) + 2 * _dot(d, _bezierQPrimePrime(bez, u));

  if (denominator == 0) {
    return u;
  } else {
    return u - numerator / denominator;
  }
}

/*
 * Assign parameter values to digitized points using relative distances between points.
 */
List<double> _chordLengthParameterize(Points points) {
  List<double> u = [];
  double currU;
  double? prevU;
  Point<double>? prevP;

  for (int i = 0; i < points.length; i+=1) {
    final p = points[i];
    currU = i > 0 ? prevU! + _norm(_sub(p, prevP!)) : 0;
    u.add(currU);

    prevU = currU;
    prevP = p;
  }

  u = u.map((x) => x / prevU!).toList();

  return u;
}


/*
 * Find the maximum squared distance of digitized points to fitted curve.
 */
_MaxError computeMaxError(Points points , Curve bez, List<num> parameters) {
  // maximum error
  double maxDist = 0;
  // index of point with maximum error
  int splitPoint = (points.length / 2).floor();

  final t_distMap = _mapTtoRelativeDistances(bez, 10);

  for (int i = 1; i < points.length - 1; i++) {
    final point = points[i];
    //Find 't' for a point on the bez curve that's as close to 'point' as possible:
    final t = _find_t(bez, parameters[i], t_distMap, 10);

    // vector from point to curve
    final v = _sub(_bezierQ(bez, t), point);
    // current error
    final dist = _squaredNorm(v);

    if (dist > maxDist) {
      maxDist = dist;
      splitPoint = i;
    }
  }

  return _MaxError(maxDist, splitPoint);
}


// Sample 't's and map them to relative distances along the curve:
List<num> _mapTtoRelativeDistances(Curve bez, num b_parts) {
  List<double> b_t_dist = [0];
  Point<double> b_t_prev = bez.point1;
  double sumLen = 0;

  for (int i = 1; i <= b_parts; i++) {
    final b_t_curr = _bezierQ(bez, i / b_parts);

    sumLen += _norm(_sub(b_t_curr, b_t_prev));

    b_t_dist.add(sumLen);
    b_t_prev = b_t_curr;
  }

  // Normalize b_length to the same interval as the parameter distances; 0 to 1:
  b_t_dist = b_t_dist.map((x) => x / sumLen).toList();
  return b_t_dist;
}

num _find_t(Curve bez, num param, List<num> t_distMap, num b_parts) {
  if (param < 0) {
    return 0;
  }
  if (param > 1) {
    return 1;
  }

  /*
        'param' is a value between 0 and 1 telling us the relative position
        of a point on the source polyline (linearly from the start (0) to the end (1)).
        To see if a given curve - 'bez' - is a close approximation of the polyline,
        we compare such a poly-point to the point on the curve that's the same
        relative distance along the curve's length.
        But finding that curve-point takes a little work:
        There is a function "B(t)" to find points along a curve from the parametric parameter 't'
        (also relative from 0 to 1: http://stackoverflow.com/a/32841764/1869660
                                    http://pomax.github.io/bezierinfo/#explanation),
        but 't' isn't linear by length (http://gamedev.stackexchange.com/questions/105230).
        So, we sample some points along the curve using a handful of values for 't'.
        Then, we calculate the length between those samples via plain euclidean distance;
        B(t) concentrates the points around sharp turns, so this should give us a good-enough outline of the curve.
        Thus, for a given relative distance ('param'), we can now find an upper and lower value
        for the corresponding 't' by searching through those sampled distances.
        Finally, we just use linear interpolation to find a better value for the exact 't'.
        More info:
            http://gamedev.stackexchange.com/questions/105230/points-evenly-spaced-along-a-bezier-curve
            http://stackoverflow.com/questions/29438398/cheap-way-of-calculating-cubic-bezier-length
            http://steve.hollasch.net/cgindex/curves/cbezarclen.html
            https://github.com/retuxx/tinyspline
    */

  // Find the two t-s that the current param distance lies between,
  // and then interpolate a somewhat accurate value for the exact t:
  for (int i = 1; i <= b_parts; i++) {
    if (param <= t_distMap[i]) {
      final tMin = (i - 1) / b_parts;
      final tMax = i / b_parts;
      final lenMin = t_distMap[i - 1];
      final lenMax = t_distMap[i];

      return ((param - lenMin) / (lenMax - lenMin)) * (tMax - tMin) + tMin;
    }
  }

  return double.nan;
}