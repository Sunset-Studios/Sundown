export function element_type_to_dom_type(element_type) {
    switch (element_type) {
        case 'panel':
            return 'div'
        case 'button':
            return 'button'
        default:
            return 'div'
    }
}

export class Element {
    name = ''
    children = []
    config = {}
    dom = null

    init(context, name, config, children = [], element_type = 'div') {
        this.name = name
        this.config = config

        this.dom = document.createElement(element_type_to_dom_type(element_type));
        this.dom.id = `${element_type}-${this.name}`;
        this.dom.classList.add(element_type);
        
        if (this.config.style) {
            Object.assign(this.dom.style, this.config.style);
        }

        if (children.length > 0) {
            children.forEach(child => {
                this.add_child(child)
            })
        }
    }

    update(delta_time) {
        this.children.forEach(child => {
            child.update(delta_time)
        })
    }

    add_child(child) {
        this.children.push(child)
        this.dom.appendChild(child.dom)
    }

    remove_child(child) {
        this.children = this.children.filter(c => c !== child)
        this.dom.removeChild(child.dom)
    }
}