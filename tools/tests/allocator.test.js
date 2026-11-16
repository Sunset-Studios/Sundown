import assert from "node:assert/strict";
import test from "node:test";
import { FreeListAllocator } from "../../engine/src/memory/allocator.js";

class AllocatedItem {
  constructor() {
    this.id = 0;
  }
}

test("free list allocator returns initialized objects after shrinking", () => {
  const allocator = new FreeListAllocator(4, AllocatedItem);
  const allocations = [];

  // Grow well beyond the initial capacity, then release most of the objects so
  // the allocator compacts its buffer.
  for (let i = 0; i < 2000; i++) allocations.push(allocator.allocate());
  for (let i = 0; i < 1990; i++) allocator.deallocate(allocations[i]);

  assert.ok(allocator.capacity < 2000, "allocator should shrink after mass deallocation");

  const available = allocator.free_list.length;
  for (let i = 0; i < available; i++) {
    const item = allocator.allocate();
    assert.ok(item instanceof AllocatedItem, "allocate() must never return undefined");
    assert.equal(item.id, 0, "recycled objects must keep their initialized fields");
  }
});
