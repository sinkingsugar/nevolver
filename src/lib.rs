use std::collections::LinkedList;

type NeuroFloat = f32;

trait Forward {
    fn forward(&mut self) -> NeuroFloat;
}

trait Backward {
    fn backward(&mut self, rate: f32, momentum: f32, update: bool, target: NeuroFloat);
}

trait Group<'a> {
    fn connect(&'a mut self, from: &'a mut Node<'a>, to: &'a mut Node<'a>);
    fn project(&'a mut self, from: &'a mut Node<'a>, to: Vec<&'a mut Node<'a>>);
}

pub enum MutationKind {
    // Node
    Activation,
    Bias,
    AddSelfConnection,
    SubSelfConnection,
    // Connection
    Weight,
    // Network
    AddNode,
    SubNode,
    AddConnection,
    SubConnection,
    AddGate,
    SubGate,
    AddBackConnection,
    SubBackConnection,
    SwapNodes,
    ShareWeight,
}

pub enum ConnectionKind {
    AllToAll,
    AllToElse,
    OneToOne,
}

pub enum GatingKind {
    Output,
    Input,
    ToSelf,
}

pub enum Activation {
    Sigmoid,
    Tanh,
    Identity,
    Relu,
    LeakyRelu,
    Step,
    Softsign,
    Sin,
    Gaussian,
    BentIdentity,
    Bipolar,
    BipolarSigmoid,
    HardTanh,
    Absolute,
    Inverse,
    Selu,
}

impl Activation {
    fn squash(self, value: NeuroFloat) -> NeuroFloat {
        match self {
            Activation::Sigmoid => unimplemented!(),
            Activation::Tanh => unimplemented!(),
            Activation::Identity => unimplemented!(),
            Activation::Relu => unimplemented!(),
            Activation::LeakyRelu => unimplemented!(),
            Activation::Step => unimplemented!(),
            Activation::Softsign => unimplemented!(),
            Activation::Sin => unimplemented!(),
            Activation::Gaussian => unimplemented!(),
            Activation::BentIdentity => unimplemented!(),
            Activation::Bipolar => unimplemented!(),
            Activation::BipolarSigmoid => unimplemented!(),
            Activation::HardTanh => unimplemented!(),
            Activation::Absolute => unimplemented!(),
            Activation::Inverse => unimplemented!(),
            Activation::Selu => unimplemented!(),
        };
    }
}

#[derive(Default, Copy, Clone)]
pub struct Connection<'a> {
    from: Option<&'a Node<'a>>,
    to: Option<&'a Node<'a>>,
    gain: NeuroFloat,
    weight_idx: Option<u32>,
    gater: Option<&'a Node<'a>>,
    elegibility: NeuroFloat,
    prev_delta_weight: NeuroFloat,
    total_delta_weight: NeuroFloat,
}

pub struct Connections<'a> {
    inbound: Vec<&'a Connection<'a>>,
    outbound: Vec<&'a Connection<'a>>,
    gated: Vec<&'a Connection<'a>>,
    to_self: Option<&'a Connection<'a>>,
}

pub enum Node<'a> {
    Input {
        value: NeuroFloat,
        connections: Connections<'a>,
    },
    Hidden {
        activation: NeuroFloat,
        connections: Connections<'a>,
        bias: NeuroFloat,
        squash: Activation,
        state: NeuroFloat,
        previous: NeuroFloat,
        mask: NeuroFloat,
        derivative: NeuroFloat,
        prev_delta_bias: NeuroFloat,
        total_delta_bias: NeuroFloat,
        output: bool,
    },
}

pub struct Network<'a> {
    nodes: LinkedList<Node<'a>>,
    connections: LinkedList<Connection<'a>>,
}

impl Forward for Node<'_> {
    fn forward(&mut self) -> NeuroFloat {
        match self {
            &mut Node::Input {
                ref value,
                connections: _,
            } => *value,
            &mut Node::Hidden {
                ref mut activation,
                ref connections,
                ref bias,
                ref squash,
                ref state,
                ref mut previous,
                ref mask,
                ref derivative,
                ref prev_delta_bias,
                ref total_delta_bias,
                ref output,
            } => {
                *previous = *state;
                *activation
            }
        }
    }
}

impl<'a> Connection<'a> {
    fn one2one(from: &'a Node<'a>, to: &'a Node<'a>) -> Connection<'a> {
        Connection {
            from: Some(from),
            to: Some(to),
            ..Default::default()
        }
    }
}

impl<'a> Group<'a> for Network<'a> {
    fn connect(&'a mut self, from: &'a mut Node<'a>, to: &'a mut Node<'a>) {
        self.connections.push_front(Connection::one2one(from, to));
        let front = self.connections.front().unwrap();
        match from {
            &mut Node::Input {
                value: _,
                ref mut connections,
            } => {
                connections.inbound.push(front);
            }
            &mut Node::Hidden {
                activation: _,
                ref mut connections,
                bias: _,
                squash: _,
                state: _,
                previous: _,
                mask: _,
                derivative: _,
                prev_delta_bias: _,
                total_delta_bias: _,
                output: _,
            } => {
                connections.inbound.push(front);
            }
        }
    }

    fn project(&mut self, from: &mut Node, to: Vec<&mut Node>) {}
}

#[cfg(test)]
mod tests {
    use crate::Connection;
    use crate::Forward;
    use crate::Node;

    #[test]
    fn node() {}
}
