import { AABB, AABB_NODE_TYPE, AABB_NODE_FLAGS } from "./aabb.js";
import { TypedQueue, TypedStack, TypedVector } from "../memory/container.js";
import { profile_scope } from "../utility/performance.js";

/**
 * Dynamic AABB Tree implementation for efficient spatial queries
 * Based on the algorithm described in "Real-Time Collision Detection" by Christer Ericson
 */
export class AABBTreeProcessor {
  // Configuration
  fat_margin_factor = 0.3; // 30% additional padding for fat AABBs
  batch_size = 512; // Process nodes in batches for better performance

  // Statistics
  leaf_nodes = 0;
  internal_nodes = 0;
  max_depth = 0;
  stats_dirty = false;

  // Tracking
  nodes_to_balance = new TypedQueue(1024);
  dirty_nodes_set = new Set();

  /**
   * Called before the update cycle
   */
  pre_update() {
    // Initialize AABB data if needed
    if (!AABB.data) AABB.initialize();
  }

  /**
   * Update the AABB tree
   * @param {number} delta_time - Time since last update
   */
  update(delta_time) {
    profile_scope("aabb_tree_processor.update", () => {
      // Process node changes
      this._process_node_changes();

      // Perform incremental balancing
      this._process_incremental_balancing();

      if (__DEV__ && this.stats_dirty) {
        // Calculate statistics
        this._calculate_stats();
      }
    });
  }

  /**
   * Called after the update cycle
   */
  post_update() {}

  /**
   * Process changes to nodes
   */
  _process_node_changes() {
    profile_scope("aabb_tree_processor.process_changes", () => {
      if (AABB.dirty_nodes.length === 0) {
        return;
      }

      const max_iterations = Math.min(AABB.dirty_nodes.length, this.batch_size);

      for (let i = 0; i < max_iterations; i++) {
        const node_index = AABB.dirty_nodes.pop();
        this.dirty_nodes_set.delete(node_index);

        const node_view = AABB.get_node_data(node_index);
        let flags = node_view.flags;
        flags &= ~AABB_NODE_FLAGS.MOVED;
        node_view.flags = flags;

        if ((flags & AABB_NODE_FLAGS.FREE) !== 0) {
          continue;
        }

        this._process_leaf_check(node_index);
      }
    });
  }

  _process_leaf_check(node_index) {
    let node_view = AABB.get_node_data(node_index);
    const min_point = node_view.min_point;
    const max_point = node_view.max_point;

    if (node_view.parent === 0 && AABB.root_node !== node_index) {
      this._process_leaf_reinsert(node_index);
      return;
    }

    const fat_aabb_min = node_view.fat_min_point;
    const fat_aabb_max = node_view.fat_max_point;

    // Node is already in the tree, check if it needs updating
    const fits_in_fat_aabb =
      min_point[0] >= fat_aabb_min[0] &&
      min_point[1] >= fat_aabb_min[1] &&
      min_point[2] >= fat_aabb_min[2] &&
      max_point[0] <= fat_aabb_max[0] &&
      max_point[1] <= fat_aabb_max[1] &&
      max_point[2] <= fat_aabb_max[2];

    if (!fits_in_fat_aabb) {
      // Remove from tree
      this._process_leaf_reinsert(node_index);
    }
  }

  _process_leaf_reinsert(node_index) {
    this._remove_leaf(node_index);

    let node_view = AABB.get_node_data(node_index);
    const min_point = node_view.min_point;
    const max_point = node_view.max_point;
    const fat_aabb = AABB.calculate_fat_margin(min_point, max_point, this.fat_margin_factor);
    node_view.fat_min_point = fat_aabb.min;
    node_view.fat_max_point = fat_aabb.max;

    this._insert_leaf(node_index);
  }

  /**
   * Insert a leaf node into the tree
   * @param {number} leaf_index - The index of the leaf node to insert
   */
  _insert_leaf(leaf_index) {
    profile_scope("aabb_tree_processor.insert_leaf", () => {
      // Skip invalid nodes
      if (leaf_index <= 0 || leaf_index >= AABB.size) return;

      let leaf_view = AABB.get_node_data(leaf_index);

      // If tree is empty, make this the root
      if (AABB.root_node === 0) {
        AABB.root_node = leaf_index;
        leaf_view.parent = 0;
        return;
      }

      // Find the best sibling for this leaf
      const sibling_index = this._find_best_sibling(leaf_index);

      // Get the old parent of the sibling
      const sibling_view = AABB.get_node_data(sibling_index);
      const old_parent = sibling_view.parent;

      // Set up the new parent
      const new_parent = this._allocate_internal_node();
      const new_parent_view = AABB.get_node_data(new_parent);
      new_parent_view.parent = old_parent;
      new_parent_view.left = sibling_index;
      new_parent_view.right = leaf_index;

      // Calculate the combined AABB
      leaf_view = AABB.get_node_data(leaf_index);

      const leaf_min_point = leaf_view.min_point;
      const leaf_max_point = leaf_view.max_point;
      const sibling_min_point = sibling_view.min_point;
      const sibling_max_point = sibling_view.max_point;

      const new_min_point = [
        Math.min(leaf_min_point[0], sibling_min_point[0]),
        Math.min(leaf_min_point[1], sibling_min_point[1]),
        Math.min(leaf_min_point[2], sibling_min_point[2]),
      ];
      const new_max_point = [
        Math.max(leaf_max_point[0], sibling_max_point[0]),
        Math.max(leaf_max_point[1], sibling_max_point[1]),
        Math.max(leaf_max_point[2], sibling_max_point[2]),
      ];

      new_parent_view.min_point = new_min_point;
      new_parent_view.max_point = new_max_point;

      // Update parent pointers
      leaf_view.parent = new_parent;
      sibling_view.parent = new_parent;

      // If sibling was the root, update root
      if (sibling_index === AABB.root_node) {
        AABB.root_node = new_parent;
        new_parent_view.parent = 0;
      } else {
        // Connect new parent to old parent
        const old_parent_view = AABB.get_node_data(old_parent);
        if (old_parent_view.left === sibling_index) {
          old_parent_view.left = new_parent;
        } else {
          old_parent_view.right = new_parent;
        }
      }

      // Refit AABBs up the tree
      this._refit_ancestors(new_parent);

      // Update the height of the new parent
      this._update_node_heights(new_parent);

      this.stats_dirty = true;
    });
  }

  /**
   * Remove a leaf node from the tree
   * @param {number} leaf_index - The index of the leaf node to remove
   */
  _remove_leaf(leaf_index) {
    profile_scope("aabb_tree_processor.remove_leaf", () => {
      const leaf_view = AABB.get_node_data(leaf_index);

      // If this is the root, clear the tree
      if (leaf_index === AABB.root_node) {
        AABB.root_node = 0;
        leaf_view.parent = 0;
        return;
      }

      const parent = leaf_view.parent;
      leaf_view.parent = 0;

      const parent_view = AABB.get_node_data(parent);
      const grandparent = parent_view.parent;

      // Find sibling
      const sibling = parent_view.left === leaf_index ? parent_view.right : parent_view.left;

      if (sibling <= 0 || sibling >= AABB.size) {
        // If parent is the root, clear the tree
        if (parent === AABB.root_node) {
          AABB.root_node = 0;
        } else if (grandparent > 0 && grandparent < AABB.size) {
          // Only disconnect from grandparent if grandparent is valid
          const grandparent_view = AABB.get_node_data(grandparent);
          if (grandparent_view.left === parent) {
            grandparent_view.left = 0;
          } else if (grandparent_view.right === parent) {
            grandparent_view.right = 0;
          }
        }

        // Free the parent node
        AABB.free_node(parent);
      } else {
        // Store sibling's parent before any modifications
        const sibling_view = AABB.get_node_data(sibling);
    
        // Connect sibling to grandparent
        if (grandparent === 0) {
          // Parent was the root, make sibling the new root
          AABB.root_node = sibling;
          sibling_view.parent = 0;
        } else {
          // Connect sibling to grandparent
          const grandparent_view = AABB.get_node_data(grandparent);
          if (grandparent_view.left === parent) {
            grandparent_view.left = sibling;
          } else {
            grandparent_view.right = sibling;
          }
          sibling_view.parent = grandparent;
        }
    
        // Free the parent node
        AABB.free_node(parent);
    
        // Refit AABBs up the tree
        this._refit_ancestors(sibling);
    
        // Update the height of the sibling
        this._update_node_heights(sibling);

        this.stats_dirty = true;
      }
    });
  }

  /**
   * Find the best sibling for a node
   * @param {number} node_index - The index of the node to find a sibling for
   * @returns {number} - The index of the best sibling
   */
  _find_best_sibling(node_index) {
    const node_view = AABB.get_node_data(node_index);

    // Start at the root
    let current = AABB.root_node;

    // If tree is empty, return 0
    if (current === 0) {
      return 0;
    }

    // Prevent infinite loops by tracking visited nodes
    const visited_nodes = new Set();

    // Calculate the cost of creating a new parent for this node and the root
    const root_view = AABB.get_node_data(current);
    const node_min_point = node_view.min_point;
    const node_max_point = node_view.max_point;
    const root_min_point = root_view.min_point;
    const root_max_point = root_view.max_point;

    const combined_aabb_min = [
      Math.min(node_min_point[0], root_min_point[0]),
      Math.min(node_min_point[1], root_min_point[1]),
      Math.min(node_min_point[2], root_min_point[2]),
    ];
    const combined_aabb_max = [
      Math.max(node_max_point[0], root_max_point[0]),
      Math.max(node_max_point[1], root_max_point[1]),
      Math.max(node_max_point[2], root_max_point[2]),
    ];

    // Calculate the surface area of the combined AABB
    const combined_surface_area = AABB.calculate_aabb_surface_area(
      combined_aabb_min,
      combined_aabb_max
    );

    // Calculate the cost of creating a new parent
    let best_cost = combined_surface_area;
    let best_sibling = current;

    // Traverse the tree to find the best sibling
    while (current !== 0) {
      // Check for cycles
      if (visited_nodes.has(current)) {
        break;
      }
      visited_nodes.add(current);

      const current_view = AABB.get_node_data(current);
      const current_min_point = current_view.min_point;
      const current_max_point = current_view.max_point;

      // If this is a leaf, it's a potential sibling
      if (current_view.node_type === AABB_NODE_TYPE.LEAF) {
        // Calculate the cost of creating a new parent for this node and the current node
        const combined_aabb_min = [
          Math.min(node_min_point[0], current_min_point[0]),
          Math.min(node_min_point[1], current_min_point[1]),
          Math.min(node_min_point[2], current_min_point[2]),
        ];
        const combined_aabb_max = [
          Math.max(node_max_point[0], current_max_point[0]),
          Math.max(node_max_point[1], current_max_point[1]),
          Math.max(node_max_point[2], current_max_point[2]),
        ];

        // Calculate the surface area of the combined AABB
        const combined_surface_area = AABB.calculate_aabb_surface_area(
          combined_aabb_min,
          combined_aabb_max
        );

        // Calculate the cost of creating a new parent
        const cost = combined_surface_area;

        // Update best sibling if this is better
        if (cost < best_cost) {
          best_cost = cost;
          best_sibling = current;
        }

        // Leaf nodes have no children, so we're done with this branch
        break;
      } else {
        // This is an internal node, calculate the cost of descending to each child
        const left = current_view.left;
        const right = current_view.right;

        // Calculate the cost of creating a new parent for this node and the current node
        const combined_aabb_min = [
          Math.min(node_min_point[0], current_min_point[0]),
          Math.min(node_min_point[1], current_min_point[1]),
          Math.min(node_min_point[2], current_min_point[2]),
        ];
        const combined_aabb_max = [
          Math.max(node_max_point[0], current_max_point[0]),
          Math.max(node_max_point[1], current_max_point[1]),
          Math.max(node_max_point[2], current_max_point[2]),
        ];

        // Calculate the surface area of the combined AABB
        const combined_surface_area = AABB.calculate_aabb_surface_area(
          combined_aabb_min,
          combined_aabb_max
        );

        // Calculate the cost of creating a new parent
        const parent_cost = combined_surface_area;

        // Update best sibling if this is better
        if (parent_cost < best_cost) {
          best_cost = parent_cost;
          best_sibling = current;
        }

        // Calculate the cost of descending to each child
        let left_cost = Infinity;
        let right_cost = Infinity;

        if (left !== 0) {
          const left_view = AABB.get_node_data(left);
          const left_min_point = left_view.min_point;
          const left_max_point = left_view.max_point;
          const left_combined_min = [
            Math.min(node_min_point[0], left_min_point[0]),
            Math.min(node_min_point[1], left_min_point[1]),
            Math.min(node_min_point[2], left_min_point[2]),
          ];
          const left_combined_max = [
            Math.max(node_max_point[0], left_max_point[0]),
            Math.max(node_max_point[1], left_max_point[1]),
            Math.max(node_max_point[2], left_max_point[2]),
          ];

          const left_surface_area = AABB.calculate_aabb_surface_area(
            left_combined_min,
            left_combined_max
          );
          left_cost =
            left_surface_area - AABB.calculate_aabb_surface_area(left_min_point, left_max_point);
        }

        if (right !== 0) {
          const right_view = AABB.get_node_data(right);
          const right_min_point = right_view.min_point;
          const right_max_point = right_view.max_point;
          const right_combined_min = [
            Math.min(node_min_point[0], right_min_point[0]),
            Math.min(node_min_point[1], right_min_point[1]),
            Math.min(node_min_point[2], right_min_point[2]),
          ];
          const right_combined_max = [
            Math.max(node_max_point[0], right_max_point[0]),
            Math.max(node_max_point[1], right_max_point[1]),
            Math.max(node_max_point[2], right_max_point[2]),
          ];

          const right_surface_area = AABB.calculate_aabb_surface_area(
            right_combined_min,
            right_combined_max
          );
          right_cost =
            right_surface_area - AABB.calculate_aabb_surface_area(right_min_point, right_max_point);
        }

        // Descend to the child with the lower cost
        if (left_cost < right_cost) {
          current = left;
        } else {
          current = right;
        }
      }
    }

    return best_sibling;
  }

  /**
   * Allocate a new internal node
   * @returns {number} - The index of the new internal node
   */
  _allocate_internal_node() {
    const node_index = AABB.allocate_node();
    const node_view = AABB.get_node_data(node_index);

    node_view.node_type = AABB_NODE_TYPE.INTERNAL;
    node_view.min_point = [-Infinity, -Infinity, -Infinity];
    node_view.max_point = [Infinity, Infinity, Infinity];
    node_view.height = 1; // Initialize height to 1

    return node_index;
  }

  /**
   * Refit the AABB of a node and its ancestors
   * @param {number} node_index - The index of the node to start refitting from
   */
  #refit_visited = new Set();
  _refit_ancestors(node_index, rebalance = true) {
    let current = node_index;

    this.#refit_visited.clear();

    while (current !== 0) {
      // Skip invalid nodes
      if (current <= 0 || current >= AABB.size) break;

      let current_view = AABB.get_node_data(current);

      // Skip leaf nodes (just move to parent)
      if (current_view.node_type === AABB_NODE_TYPE.LEAF) {
        current = current_view.parent;
        continue;
      }

      // Check for cycles
      if (this.#refit_visited.has(current)) {
        // Fix the cycle by breaking the parent link
        current_view.parent = 0;
        break; // Exit the loop to prevent infinite cycling
      }

      this.#refit_visited.add(current);

      // Skip free nodes
      if ((current_view.flags & AABB_NODE_FLAGS.FREE) !== 0) break;

      // Get children
      const left = current_view.left;
      const right = current_view.right;

      // Validate children to prevent self-references
      if (left === current || right === current) {
        // Fix self-reference by clearing the child
        if (left === current) current_view.left = 0;
        if (right === current) current_view.right = 0;
        current = current_view.parent;
        continue;
      }

      // Start with infinite bounds
      let min_point = [Infinity, Infinity, Infinity];
      let max_point = [-Infinity, -Infinity, -Infinity];

      // Include left child if valid
      if (left !== 0) {
        const left_view = AABB.get_node_data(left);
        const left_view_min = left_view.min_point;
        const left_view_max = left_view.max_point;

        // Skip if left child is free
        if ((left_view.flags & AABB_NODE_FLAGS.FREE) === 0) {
          min_point[0] = Math.min(min_point[0], left_view_min[0]);
          min_point[1] = Math.min(min_point[1], left_view_min[1]);
          min_point[2] = Math.min(min_point[2], left_view_min[2]);
          max_point[0] = Math.max(max_point[0], left_view_max[0]);
          max_point[1] = Math.max(max_point[1], left_view_max[1]);
          max_point[2] = Math.max(max_point[2], left_view_max[2]);
        }
      }

      // Include right child if valid
      if (right !== 0) {
        const right_view = AABB.get_node_data(right);
        const right_view_min = right_view.min_point;
        const right_view_max = right_view.max_point;

        // Skip if right child is free
        if ((right_view.flags & AABB_NODE_FLAGS.FREE) === 0) {
          min_point[0] = Math.min(min_point[0], right_view_min[0]);
          min_point[1] = Math.min(min_point[1], right_view_min[1]);
          min_point[2] = Math.min(min_point[2], right_view_min[2]);
          max_point[0] = Math.max(max_point[0], right_view_max[0]);
          max_point[1] = Math.max(max_point[1], right_view_max[1]);
          max_point[2] = Math.max(max_point[2], right_view_max[2]);
        }
      }

      // Update bounds
      current_view.min_point = min_point;
      current_view.max_point = max_point;

      // Move to parent
      const parent = current_view.parent;

      // Check for parent-child cycle
      if (parent === current) {
        current_view.parent = 0; // Break the cycle
        break;
      }

      if (rebalance) {
        this.nodes_to_balance.push(current);
      }

      current = parent;
    }
  }

  /**
   * Process incremental balancing
   */
  _process_incremental_balancing() {
    profile_scope("aabb_tree_processor.balance", () => {
      // Skip if no nodes to balance
      if (this.nodes_to_balance.length <= 0) {
        return;
      }

      // Process a limited number of nodes per update
      const num_to_process = Math.min(this.nodes_to_balance.length, this.batch_size);

      for (let i = 0; i < num_to_process; i++) {
        const node_index = this.nodes_to_balance.pop();

        // Skip invalid nodes
        if (node_index <= 0 || node_index >= AABB.size) continue;

        const node_view = AABB.get_node_data(node_index);

        // Skip non-internal nodes
        if (node_view.node_type !== AABB_NODE_TYPE.INTERNAL) continue;

        // Balance this node
        this._balance_node(node_index);
      }
    });
  }

  /**
   * Balance a node
   * @param {number} node_index - The index of the node to balance
   */
  _balance_node(node_index) {
    const node_view = AABB.get_node_data(node_index);

    // Get children
    const left = node_view.left;
    const right = node_view.right;

    // Skip if no children
    if (left === 0 && right === 0) return;

    // Get the height of each subtree using cached values
    const left_height = left <= 0 ? 0 : this._get_node_height(left);
    const right_height = right <= 0 ? 0 : this._get_node_height(right);

    // Check if the tree is balanced
    const height_diff = Math.abs(left_height - right_height);

    if (height_diff <= 1) {
      // Tree is balanced
      return;
    }

    // Tree is unbalanced, rotate
    if (left_height > right_height) {
      // Left subtree is taller, rotate right
      this._rotate_right(node_index);
    } else {
      // Right subtree is taller, rotate left
      this._rotate_left(node_index);
    }
  }

  /**
   * Get the cached height of a node or calculate it if not available
   * @param {number} node_index - The index of the node
   * @returns {number} - The height of the node
   * @private
   */
  _get_node_height(node_index) {
    // Skip invalid nodes
    if (node_index <= 0 || node_index >= AABB.size) return 0;

    const node_view = AABB.get_node_data(node_index);

    return node_view.height || 0;
  }

  /**
   * Rotate a subtree left
   * @param {number} node_index - The index of the root of the subtree
   */
  _rotate_left(node_index) {
    const node_view = AABB.get_node_data(node_index);

    // Skip if not an internal node
    if (node_view.node_type !== AABB_NODE_TYPE.INTERNAL) return;

    // Get right child
    const right = node_view.right;

    // Skip if no right child or if right child is the node itself
    if (right === 0 || right === node_index) return;

    const right_view = AABB.get_node_data(right);

    // Skip if right child is not an internal node
    if (right_view.node_type !== AABB_NODE_TYPE.INTERNAL) return;

    // Get right child's children
    const right_left = right_view.left;
    const right_right = right_view.right;

    // Skip if right child has no children or if any child is the node itself
    if (right_left === 0 && right_right === 0) return;
    if (right_left === node_index || right_right === node_index) return;

    // Store original parent before any modifications
    const parent = node_view.parent;

    // Skip if parent is the right child (would create a cycle)
    if (parent === right) return;

    // Update parent pointers
    right_view.parent = parent;
    node_view.parent = right;

    // Update child pointers
    right_view.left = node_index;
    node_view.right = right_left;

    // Update right_left's parent pointer
    if (right_left !== 0) {
      const right_left_view = AABB.get_node_data(right_left);
      right_left_view.parent = node_index;
    }

    // Update parent's child pointer
    if (parent === 0) {
      // Node was the root, update root
      AABB.root_node = right;
    } else {
      // Update parent's child pointer
      const parent_view = AABB.get_node_data(parent);
      if (parent_view.left === node_index) {
        parent_view.left = right;
      } else {
        parent_view.right = right;
      }
    }

    // Refit the ancestors without rebalancing
    this._refit_ancestors(node_index, false);

    // Update heights
    this._update_node_heights(node_index);
  }

  /**
   * Rotate a subtree right
   * @param {number} node_index - The index of the root of the subtree
   */
  _rotate_right(node_index) {
    const node_view = AABB.get_node_data(node_index);

    // Skip if not an internal node
    if (node_view.node_type !== AABB_NODE_TYPE.INTERNAL) return;

    // Get left child
    const left = node_view.left;

    // Skip if no left child or if left child is the node itself
    if (left === 0 || left === node_index) return;

    const left_view = AABB.get_node_data(left);

    // Skip if left child is not an internal node
    if (left_view.node_type !== AABB_NODE_TYPE.INTERNAL) return;

    // Get left child's children
    const left_left = left_view.left;
    const left_right = left_view.right;

    // Skip if left child has no children or if any child is the node itself
    if (left_left === 0 && left_right === 0) return;
    if (left_left === node_index || left_right === node_index) return;

    // Store original parent before any modifications
    const parent = node_view.parent;

    // Skip if parent is the left child (would create a cycle)
    if (parent === left) return;

    // Update parent pointers
    left_view.parent = parent;
    node_view.parent = left;

    // Update child pointers
    left_view.right = node_index;
    node_view.left = left_right;

    // Update left_right's parent pointer
    if (left_right !== 0) {
      const left_right_view = AABB.get_node_data(left_right);
      left_right_view.parent = node_index;
    }

    // Update parent's child pointer
    if (parent === 0) {
      // Node was the root, update root
      AABB.root_node = left;
    } else {
      // Update parent's child pointer
      const parent_view = AABB.get_node_data(parent);
      if (parent_view.left === node_index) {
        parent_view.left = left;
      } else {
        parent_view.right = left;
      }
    }

    // Refit the ancestors without rebalancing
    this._refit_ancestors(node_index, false);

    // Update heights
    this._update_node_heights(node_index);
  }

  /**
   * Calculate statistics about the tree
   */
  _calculate_stats() {
    profile_scope("aabb_tree_processor.calculate_stats", () => {
      // Skip if tree is empty
      if (AABB.root_node === 0) {
        return;
      }

      // Count nodes
      this._count_nodes(AABB.root_node);

      // Update depth using cached height values
      this._update_node_depth(AABB.root_node);

      // Mark stats as not dirty
      this.stats_dirty = false;
    });
  }

  /**
   * Update the maximum depth of the tree
   * @param {number} start_node_index - The index of the node to start from
   */
  _update_node_depth(start_node_index) {
    // Skip invalid nodes
    if (start_node_index <= 0 || start_node_index >= AABB.size) return;

    // Get the height of the tree using cached values
    const tree_height = this._get_node_height(start_node_index);

    // Update max depth statistic
    this.max_depth = tree_height - 1; // Height - 1 = depth (root is at depth 0)
  }

  /**
   * Count nodes and update depths using an iterative approach
   * @param {number} start_node_index - The index of the node to start from
   * @param {number} start_depth - The depth of the start node
   */
  #count_nodes_stack = new TypedStack(1024, Uint32Array);
  #count_nodes_visited = new Set();
  _count_nodes(start_node_index) {
    // Skip if tree is empty
    if (start_node_index <= 0 || start_node_index >= AABB.size) return;

    // reset the number of nodes
    this.leaf_nodes = 0;
    this.internal_nodes = 0;

    // Use a stack to track nodes to process
    this.#count_nodes_stack.clear();
    this.#count_nodes_visited.clear();

    this.#count_nodes_stack.push(start_node_index);

    while (this.#count_nodes_stack.length > 0) {
      const node_index = this.#count_nodes_stack.pop();

      if (this.#count_nodes_visited.has(node_index)) continue;

      this.#count_nodes_visited.add(node_index);

      // Skip invalid nodes
      if (node_index <= 0 || node_index >= AABB.size) continue;

      const node_view = AABB.get_node_data(node_index);

      // Count this node
      if (node_view.node_type === AABB_NODE_TYPE.LEAF) {
        this.leaf_nodes++;
      } else if (node_view.node_type === AABB_NODE_TYPE.INTERNAL) {
        this.internal_nodes++;
      }

      // Add children to stack if this is an internal node
      if (node_view.node_type === AABB_NODE_TYPE.INTERNAL) {
        // Push right child first so left child is processed first (stack is LIFO)
        if (node_view.right > 0) {
          this.#count_nodes_stack.push(node_view.right);
        }

        if (node_view.left > 0) {
          this.#count_nodes_stack.push(node_view.left);
        }
      }
    }
  }

  /**
   * Mark a node as dirty
   * @param {number} node_index - The index of the node to mark as dirty
   */
  mark_node_dirty(node_index) {
    if (node_index <= 0 || node_index >= AABB.size) return;

    if (!this.dirty_nodes_set.has(node_index)) {
      this.dirty_nodes_set.add(node_index);
      AABB.dirty_nodes.push(node_index);
    }
  }

  /**
   * Remove a node from the tree
   * @param {number} node_index - The index of the node to remove
   */
  remove_node_from_tree(node_index) {
    // Skip invalid nodes
    if (node_index <= 0 || node_index >= AABB.size) return;

    const node_view = AABB.get_node_data(node_index);

    // Skip free nodes
    if ((node_view.flags & AABB_NODE_FLAGS.FREE) !== 0) return;

    // Remove from tree
    this._remove_leaf(node_index);
  }

  /**
   * Get statistics about the tree
   * @returns {Object} - Statistics about the tree
   */
  get_stats() {
    return {
      allocated_nodes: AABB.allocated_count,
      leaf_nodes: this.leaf_nodes,
      internal_nodes: this.internal_nodes,
      max_depth: this.max_depth,
      nodes_to_balance: this.nodes_to_balance.length,
      dirty_nodes: AABB.dirty_nodes.length,
    };
  }

  /**
   * Update the height of a node and its ancestors
   * @param {number} node_index - The index of the node to start updating from
   * @private
   */
  #update_node_heights_visited = new Set();
  _update_node_heights(node_index) {
    let current = node_index;
    this.#update_node_heights_visited.clear();

    // Add a safety counter to prevent infinite loops

    while (current > 0) {
      // Check for cycles
      if (this.#update_node_heights_visited.has(current)) {
        const node_view = AABB.get_node_data(current);
        node_view.parent = 0; // Break the cycle
        break;
      }

      this.#update_node_heights_visited.add(current);

      const node_view = AABB.get_node_data(current);

      // Skip free nodes
      if ((node_view.flags & AABB_NODE_FLAGS.FREE) !== 0) break;

      // Calculate new height
      let new_height = 0;

      if (node_view.node_type === AABB_NODE_TYPE.LEAF) {
        new_height = 1;
      } else if (node_view.node_type === AABB_NODE_TYPE.INTERNAL) {
        // Directly access child heights without recursion
        const left_child = node_view.left;
        const right_child = node_view.right;

        // Validate children to prevent self-references
        if (left_child === current) node_view.left = 0;
        if (right_child === current) node_view.right = 0;

        const left_height =
          left_child <= 0 || left_child === current
            ? 0
            : AABB.get_node_data(left_child).height || 1;
        const right_height =
          right_child <= 0 || right_child === current
            ? 0
            : AABB.get_node_data(right_child).height || 1;

        new_height = Math.max(left_height, right_height) + 1;
      }

      // If height hasn't changed, we can stop propagating updates
      if (node_view.height === new_height) break;

      // Update the height
      node_view.height = new_height;

      // Move to parent
      const parent = node_view.parent;

      // Check for parent-child cycle
      if (parent === current) {
        node_view.parent = 0; // Break the cycle
        break;
      }

      current = parent;
    }
  }
}
