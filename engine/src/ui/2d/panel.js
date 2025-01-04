import { Element } from './element.js'

export class Panel extends Element {
    static create(name, config, children = []) {
        const panel = new Panel()
        panel.init(name, config, children, 'panel')
        return panel
    }
}