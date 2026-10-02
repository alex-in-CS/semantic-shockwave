// Sound effects, synthesized with WebAudio (no files to load). Off by default;
// the choice is remembered per browser.

const KEY = 'shockwave.sound';
let ctx = null;
let enabled = false;
try { enabled = localStorage.getItem(KEY) === 'on'; } catch { /* storage unavailable */ }

function audio() {
  if (!enabled) return null;
  ctx ??= new AudioContext();
  if (ctx.state === 'suspended') ctx.resume();
  return ctx;
}

export const isSoundOn = () => enabled;

export function setSound(on) {
  enabled = on;
  try { localStorage.setItem(KEY, on ? 'on' : 'off'); } catch { /* storage unavailable */ }
  if (on) audio();
}

function noiseBuffer(ac, seconds) {
  const buffer = ac.createBuffer(1, ac.sampleRate * seconds, ac.sampleRate);
  const data = buffer.getChannelData(0);
  for (let i = 0; i < data.length; i += 1) data[i] = Math.random() * 2 - 1;
  return buffer;
}

function envelope(ac, gain, attack, peak, release, when = ac.currentTime) {
  const g = ac.createGain();
  g.gain.setValueAtTime(0.0001, when);
  g.gain.exponentialRampToValueAtTime(peak * gain, when + attack);
  g.gain.exponentialRampToValueAtTime(0.0001, when + attack + release);
  g.connect(ac.destination);
  return g;
}

/** Rising filtered-noise sweep for the comets. */
export function whoosh(seconds = 1.1) {
  const ac = audio();
  if (!ac) return;
  const src = ac.createBufferSource();
  src.buffer = noiseBuffer(ac, seconds);
  const filter = ac.createBiquadFilter();
  filter.type = 'bandpass';
  filter.Q.value = 1.2;
  filter.frequency.setValueAtTime(300, ac.currentTime);
  filter.frequency.exponentialRampToValueAtTime(3200, ac.currentTime + seconds);
  src.connect(filter).connect(envelope(ac, 0.35, seconds * 0.85, 1, seconds * 0.15));
  src.start();
}

/** Low thump plus a noise burst for the blast. */
export function boom() {
  const ac = audio();
  if (!ac) return;
  const osc = ac.createOscillator();
  osc.type = 'sine';
  osc.frequency.setValueAtTime(120, ac.currentTime);
  osc.frequency.exponentialRampToValueAtTime(32, ac.currentTime + 0.9);
  osc.connect(envelope(ac, 0.9, 0.01, 1, 1.1));
  osc.start();
  osc.stop(ac.currentTime + 1.2);
  const src = ac.createBufferSource();
  src.buffer = noiseBuffer(ac, 1.2);
  const filter = ac.createBiquadFilter();
  filter.type = 'lowpass';
  filter.frequency.setValueAtTime(2400, ac.currentTime);
  filter.frequency.exponentialRampToValueAtTime(120, ac.currentTime + 1.1);
  src.connect(filter).connect(envelope(ac, 0.5, 0.005, 1, 1.0));
  src.start();
}

let lastTick = 0;
/** A soft blip while a search explores; throttled so it doesn't buzz. */
export function tick(pitch = 0) {
  const ac = audio();
  if (!ac || ac.currentTime - lastTick < 0.045) return;
  lastTick = ac.currentTime;
  const osc = ac.createOscillator();
  osc.type = 'triangle';
  osc.frequency.value = 520 + pitch * 380;
  osc.connect(envelope(ac, 0.05, 0.004, 1, 0.06));
  osc.start();
  osc.stop(ac.currentTime + 0.08);
}

/** A chime for each hop of the tour, rising along the path. */
export function chime(step = 0) {
  const ac = audio();
  if (!ac) return;
  const notes = [523.25, 587.33, 659.25, 783.99, 880, 987.77, 1046.5];
  const osc = ac.createOscillator();
  osc.type = 'sine';
  osc.frequency.value = notes[step % notes.length];
  osc.connect(envelope(ac, 0.16, 0.01, 1, 0.9));
  osc.start();
  osc.stop(ac.currentTime + 1);
}
