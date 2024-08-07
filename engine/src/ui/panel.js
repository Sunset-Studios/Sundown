import { Element } from './element.js'

export class Panel extends Element {
    static create(context, name, config, children = []) {
        const panel = new Panel()
        panel.init(context, name, config, children, 'panel')
        return panel
    }
}