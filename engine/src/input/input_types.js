/**
 * Enumeration of input keys.
 * @enum {number}
 */
export const InputKey = {
    K_None: 0,
    K_Return: 1,
    K_Escape: 2,
    K_Backspace: 3,
    K_Tab: 4,
    K_Space: 5,
    K_Exclaim: 6,
    K_Quotedbl: 7,
    K_Hash: 8,
    K_Percent: 9,
    K_Dollar: 10,
    K_Ampersand: 11,
    K_Quote: 12,
    K_Leftparen: 13,
    K_Rightparen: 14,
    K_Asterisk: 15,
    K_Plus: 16,
    K_Comma: 17,
    K_Minus: 18,
    K_Period: 19,
    K_Slash: 20,
    K_Colon: 21,
    K_Semicolon: 22,
    K_Less: 23,
    K_Equals: 24,
    K_Greater: 25,
    K_Question: 26,
    K_At: 27,
    K_Leftbracket: 28,
    K_Backslash: 29,
    K_Rightbracket: 30,
    K_Caret: 31,
    K_Underscore: 32,
    K_Backquote: 33,
    K_0: 34,
    K_1: 35,
    K_2: 36,
    K_3: 37,
    K_4: 38,
    K_5: 39,
    K_6: 40,
    K_7: 41,
    K_8: 42,
    K_9: 43,
    K_a: 44,
    K_b: 45,
    K_c: 46,
    K_d: 47,
    K_e: 48,
    K_f: 49,
    K_g: 50,
    K_h: 51,
    K_i: 52,
    K_j: 53,
    K_k: 54,
    K_l: 55,
    K_m: 56,
    K_n: 57,
    K_o: 58,
    K_p: 59,
    K_q: 60,
    K_r: 61,
    K_s: 62,
    K_t: 63,
    K_u: 64,
    K_v: 65,
    K_w: 66,
    K_x: 67,
    K_y: 68,
    K_z: 69,
    K_LShift: 70,
    K_RShift: 71,
    K_LControl: 72,
    K_RControl: 73,
    K_LAlt: 74,
    K_RAlt: 75,
    K_Up: 76,
    K_Down: 77,
    K_Left: 78,
    K_Right: 79,
    B_mouse_left: 80,
    B_mouse_right: 81,
    B_mouse_middle: 82,
    NumKeys: 83
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

export const InputKeyToPrintableString = {
    [InputKey.K_Return]: "Return",
    [InputKey.K_Escape]: "Escape",
    [InputKey.K_Backspace]: "Backspace",
    [InputKey.K_Tab]: "\t",
    [InputKey.K_Space]: " ",
    [InputKey.K_Exclaim]: "!",
    [InputKey.K_Quotedbl]: "\"",
    [InputKey.K_Hash]: "#",
    [InputKey.K_Percent]: "%",
    [InputKey.K_Dollar]: "$",
    [InputKey.K_Ampersand]: "&",
    [InputKey.K_Quote]: "'",
    [InputKey.K_Leftparen]: "(",
    [InputKey.K_Rightparen]: ")",
    [InputKey.K_Asterisk]: "*",
    [InputKey.K_Plus]: "+",
    [InputKey.K_Comma]: ",",
    [InputKey.K_Minus]: "-",
    [InputKey.K_Period]: ".",
    [InputKey.K_Slash]: "/",
    [InputKey.K_Colon]: ":",
    [InputKey.K_Semicolon]: ";",
    [InputKey.K_Less]: "<",
    [InputKey.K_Equals]: "=",
    [InputKey.K_Greater]: ">",
    [InputKey.K_Question]: "?",
    [InputKey.K_At]: "@",
    [InputKey.K_Leftbracket]: "[",
    [InputKey.K_Backslash]: "\\",
    [InputKey.K_Rightbracket]: "]",
    [InputKey.K_Caret]: "^",
    [InputKey.K_Underscore]: "_",
    [InputKey.K_Backquote]: "`",
    [InputKey.K_0]: "0",
    [InputKey.K_1]: "1",
    [InputKey.K_2]: "2",
    [InputKey.K_3]: "3",
    [InputKey.K_4]: "4",
    [InputKey.K_5]: "5",
    [InputKey.K_6]: "6",
    [InputKey.K_7]: "7",
    [InputKey.K_8]: "8",
    [InputKey.K_9]: "9",
    [InputKey.K_a]: "a",
    [InputKey.K_b]: "b",
    [InputKey.K_c]: "c",
    [InputKey.K_d]: "d",
    [InputKey.K_e]: "e",
    [InputKey.K_f]: "f",
    [InputKey.K_g]: "g",
    [InputKey.K_h]: "h",
    [InputKey.K_i]: "i",
    [InputKey.K_j]: "j",
    [InputKey.K_k]: "k",
    [InputKey.K_l]: "l",
    [InputKey.K_m]: "m",
    [InputKey.K_n]: "n",
    [InputKey.K_o]: "o",
    [InputKey.K_p]: "p",
    [InputKey.K_q]: "q",
    [InputKey.K_r]: "r",
    [InputKey.K_s]: "s",
    [InputKey.K_t]: "t",
    [InputKey.K_u]: "u",
    [InputKey.K_v]: "v",
    [InputKey.K_w]: "w",
    [InputKey.K_x]: "x",
    [InputKey.K_y]: "y",
    [InputKey.K_z]: "z",
    [InputKey.K_LShift]: "",
    [InputKey.K_RShift]: "",
    [InputKey.K_LControl]: "",
    [InputKey.K_RControl]: "",
    [InputKey.K_LAlt]: "",
    [InputKey.K_RAlt]: "",
    [InputKey.B_mouse_left]: "",
    [InputKey.B_mouse_right]: "",
    [InputKey.B_mouse_middle]: "",
    [InputKey.K_Up]: "",
    [InputKey.K_Down]: "",
    [InputKey.K_Left]: "",
    [InputKey.K_Right]: "",
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
        this.last_change_time = 0.0;
    }
}
