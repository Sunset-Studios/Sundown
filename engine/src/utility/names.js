export class Name {
    static string_to_hash = new Map();
    static hash_to_string = new Map();

    constructor(str) {
        if (Name.string_to_hash.has(str)) {
            this.hash = Name.string_to_hash.get(str);
        } else {
            this.hash = Name.fnv1a_hash(str);
            Name.string_to_hash.set(str, this.hash);
            Name.hash_to_string.set(this.hash, str);
        }
    }

    static fnv1a_hash(str) {
        let hash = 0x811c9dc5;
        for (let i = 0; i < str.length; i++) {
            hash ^= str.charCodeAt(i);
            hash = (hash * 0x01000193) >>> 0; // Force 32-bit unsigned integer
        }
        return hash;
    }

    static string(hash) {
        return Name.hash_to_string.get(hash);
    }

    static hash(str) {
        return Name.string_to_hash.get(str);
    }

    static from(str) {
        const name = new Name(str);
        return name.hash;
    }
}