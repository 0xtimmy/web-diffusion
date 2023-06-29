export type Shape = number[];
export type Shapeish = Shape | number;
export type Strides = number[];

export function getShape(shapeish: Shapeish): Shape {
    if (typeof shapeish === "number") {
        return [shapeish];
    } else {
        return shapeish;
    }
}

export function shapeSize(shape: Shape): number {
    if (shape.length === 0) {
        return 1;
    }
    else if (shape.length === 1) {
        return shape[0];
    }
    else {
        return shape.reduce((a, b) => a * b);
    }
}

export function defaultStrides(shape: Shape): Strides {
    const strides = [];
    let stride = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

export function shapesEq(a: Shape, b: Shape): boolean {
    if(a.length != b.length) return false;
    return a.reduce((acc, v, i) => {
        return acc && v == b[i];
    }, true);
}
