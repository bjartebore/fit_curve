
Iterable<R> _indexedMap<T, R>(Iterable<T> list, R Function(T val, int index, Iterable<T> list) f) sync* {
  int i = 0;
  for (final T val in list) {
    yield f(val, i++, list);
  }
}

extension IndexedMap<E> on List<E> {

  Iterable<R> indexedMap<R>(R Function(E val, int index, Iterable<E> list) f) sync* {
    yield* _indexedMap(this, f);
  }

  List<E> slice(int begin, [int? end]) => getRange(
        begin, end == null ? length : end < 0 ? length + end : end)
    .toList();

}
