const fs = require("fs"),
  vm = require("vm"),
  assert = require("assert");
const context = vm.createContext({});
const source = fs.readFileSync(
  require("path").join(__dirname, "../memory_report.js"),
  "utf8",
);
vm.runInContext(source + ";globalThis.report=MemoryReport;", context);
const data = {
  kind: "runtime_memory",
  name: "fixture",
  replay: JSON.parse(fs.readFileSync(process.argv[2], "utf8")),
};
context.report.validate(data);
const bad = JSON.parse(JSON.stringify(data));
bad.replay.profiles[0].peak_stacks[0].bytes++;
assert.throws(() => context.report.validate(bad), /conserve/);
const duplicate = JSON.parse(JSON.stringify(data));
duplicate.replay.profiles.push(duplicate.replay.profiles[0]);
assert.throws(() => context.report.validate(duplicate), /Duplicate/);
const timeline = JSON.parse(JSON.stringify(data));
timeline.replay.profiles[0].points[0][0] = NaN;
assert.throws(() => context.report.validate(timeline), /timeline/);
console.log("Memory report validation contracts passed");

for (const patch of [
  { new_active_peak_bytes: -1 },
  { reconstructed_peak_bytes: -1 },
  { stage_timeline: [{ stage: "DiT", start: "0", end: 1 }] },
  { dit_calls: [{ phase: "denoise", start: 2, end: 1 }] },
]) {
  const invalid = JSON.parse(JSON.stringify(data));
  Object.assign(invalid.replay.profiles[0], patch);
  assert.throws(() => context.report.validate(invalid), /peaks|stage timeline/);
}
