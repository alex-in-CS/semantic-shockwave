/** Decode the backend's int8/base64 vectors into normalized Float32Arrays. */
export function decodeVectors({ data, dims, scale }, count) {
  const bytes = Uint8Array.from(atob(data), (c) => c.charCodeAt(0));
  const raw = new Int8Array(bytes.buffer);
  const vectors = [];
  for (let i = 0; i < count; i += 1) {
    const v = Float32Array.from(raw.subarray(i * dims, (i + 1) * dims), (x) => x / scale);
    let norm = 0;
    for (let k = 0; k < dims; k += 1) norm += v[k] * v[k];
    norm = Math.sqrt(norm) || 1;
    for (let k = 0; k < dims; k += 1) v[k] /= norm;
    vectors.push(v);
  }
  return vectors;
}
