/**
 * Enumeration of input keys.
 * @enum {number}
 */
export const InputKey = {
    K_Return: 0,
    K_Escape: 1,
    K_Backspace: 2,
    K_Tab: 3,
    K_Space: 4,
    K_Exclaim: 5,
    K_Quotedbl: 6,
    K_Hash: 7,
    K_Percent: 8,
    K_Dollar: 9,
    K_Ampersand: 10,
    K_Quote: 11,
    K_Leftparen: 12,
    K_Rightparen: 13,
    K_Asterisk: 14,
    K_Plus: 15,
    K_Comma: 16,
    K_Minus: 17,
    K_Period: 18,
    K_Slash: 19,
    K_Colon: 20,
    K_Semicolon: 21,
    K_Less: 22,
    K_Equals: 23,
    K_Greater: 24,
    K_Question: 25,
    K_At: 26,
    K_Leftbracket: 27,
    K_Backslash: 28,
    K_Rightbracket: 29,
    K_Caret: 30,
    K_Underscore: 31,
    K_Backquote: 32,
    K_0: 33,
    K_1: 34,
    K_2: 35,
    K_3: 36,
    K_4: 37,
    K_5: 38,
    K_6: 39,
    K_7: 40,
    K_8: 41,
    K_9: 42,
    K_a: 43,
    K_b: 44,
    K_c: 45,
    K_d: 46,
    K_e: 47,
    K_f: 48,
    K_g: 49,
    K_h: 50,
    K_i: 51,
    K_j: 52,
    K_k: 53,
    K_l: 54,
    K_m: 55,
    K_n: 56,
    K_o: 57,
    K_p: 58,
    K_q: 59,
    K_r: 60,
    K_s: 61,
    K_t: 62,
    K_u: 63,
    K_v: 64,
    K_w: 65,
    K_x: 66,
    K_y: 67,
    K_z: 68,
    K_LShift: 69,
    K_RShift: 70,
    B_mouse_left: 71,
    B_mouse_right: 72,
    B_mouse_middle: 71,
    NumKeys: 72
};

/**
 * Enumeration of input ranges.
 * @enum {number}
 */
export const InputRange = {
    M_x: 0,
    M_y: 1,
    M_xabs: 2,
    M_yabs: 3,
    M_wheel: 4,
    NumRanges: 5
};

/**
 * Enumeration of input types.
 * @enum {number}
 */
export const InputType = {
    Action: 0,
    State: 1,
    Range: 2
};

/**
 * Represents the state of an input.
 * @class
 * @property {string} mapped_name - The mapped name of the input.
 * @property {InputKey} raw_input - The raw input key.
 * @property {InputRange} raw_range - The raw input range.
 * @property {InputType} input_type - The type of input.
 * @property {number} range_value - The value of the range input (default: 0.0).
 */
export class InputState {
    constructor(mapped_name, raw_input, raw_range, input_type) {
        this.mapped_name = mapped_name;
        this.raw_input = raw_input;
        this.raw_range = raw_range;
        this.input_type = input_type;
        this.range_value = 0.0;
    }
}
