use std::rc::{Rc, Weak};
use std::cell::Cell;

type NeuroFloat = f32;
type NodeRef = Rc<Cell<Node>>;
type ConnRef = Rc<Cell<Connection>>;

trait Forward {
    fn forward(&mut self) -> NeuroFloat;
}

trait Backward {
    fn backward(&mut self, rate: f32, momentum: f32, update: bool, target: NeuroFloat);
}

trait Group {
    fn connect(&mut self, from: &mut NodeRef, to: NodeRef);
    fn project(&mut self, from: NodeRef, to: Vec<NodeRef>);
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

#[derive(Default)]
pub struct Connection {
    from: Option<NodeRef>,
    to: Option<NodeRef>,
    gain: NeuroFloat,
    weight: Rc<NeuroFloat>,
    gater: Option<NodeRef>,
    elegibility: NeuroFloat,
    prev_delta_weight: NeuroFloat,
    total_delta_weight: NeuroFloat,
}

#[derive(Default)]
pub struct Connections {
    outbound: Vec<Connection>,
    inbound: Vec<ConnRef>,
    gated: Vec<ConnRef>,
    to_self: Box<Connection>,
}

pub enum Node {
    Input {
        value: NeuroFloat,
        connections: Connections,
    },
    Hidden {
        activation: NeuroFloat,
        connections: Connections,
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

pub struct Network {
    nodes: Vec<NodeRef>,
    connections: Vec<ConnRef>,
}

impl Forward for Node {
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

impl Connection {
    fn one2one(from: NodeRef, to: NodeRef) -> Connection {
        Connection {
            from: Some(from),
            to: Some(to),
            weight: Rc::<NeuroFloat>::new(0.0),
            ..Default::default()
        }
    }
}

impl Group for Network {
    fn connect(&mut self, from: &mut NodeRef, to: NodeRef) {
        let conn = Cell::new(Connection::one2one(from.clone(), to));
        let conn_ref = Rc::<Cell<Connection>>::new(conn);
        self.connections.push(conn_ref.clone());
        let from_node = Rc::get_mut(from).unwrap().get_mut();
        match from_node {
            Node::Input {
                value: _,
                ref mut connections,
            } => {
                connections.inbound.push(conn_ref.clone());
            }
            Node::Hidden {
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
                connections.inbound.push(conn_ref.clone());
            }
        }
    }

    fn project(&mut self, from: NodeRef, to: Vec<NodeRef>) {}
}

#[cfg(test)]
mod tests {
    use crate::Network;
use crate::Connection;
    use crate::Connections;
    use crate::Forward;
    use crate::Node;
    use crate::Group;

    #[test]
    fn node() {
        let input = Node::Input{
            value: 0.0,
            connections: Connections::default()
        };

        let mut network = Network{
            nodes: vec![],
            connections: vec![]
        };

        // network.connect(input, input);
    }
}
