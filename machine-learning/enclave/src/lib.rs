// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License..

#![crate_name = "machinelearningsampleenclave"]
#![crate_type = "staticlib"]

#![cfg_attr(not(target_env = "sgx"), no_std)]
#![cfg_attr(target_env = "sgx", feature(rustc_private))]

extern crate sgx_types;
#[cfg(not(target_env = "sgx"))]
#[macro_use]
extern crate sgx_tstd as std;
extern crate tensorflow;

use sgx_types::*;
use std::vec::Vec;
use std::time::*;
use std::untrusted::time::SystemTimeEx;
use std::path::Path;
use std::error::Error;

use tensorflow::*;
use tensorflow::ops;
use tensorflow::train::*;
use tensorflow::train::util::*;

#[no_mangle]
pub extern "C"
fn sample_main() -> sgx_status_t {
    // Run our TensorFlow XOR example
    match train_xor_model() {
        Ok(_) => println!("Training completed successfully"),
        Err(e) => println!("Training failed: {:?}", e)
    }

    sgx_status_t::SGX_SUCCESS
}

fn train_xor_model() -> Result<(), Box<dyn Error>> {
    // Create a new TensorFlow scope
    let mut scope = Scope::new_root_scope();
    let scope = &mut scope;

    // Define the model architecture
    let input = ops::Placeholder::new()
        .dtype(DataType::Float)
        .shape([1u64, 2])
        .build(&mut scope.with_op_name("input"))?;
    
    let label = ops::Placeholder::new()
        .dtype(DataType::Float)
        .shape([1u64])
        .build(&mut scope.with_op_name("label"))?;

    // Create hidden layer
    let hidden_size: u64 = 8;
    let (vars1, layer1) = layer(
        input.clone(),
        2,
        hidden_size,
        &|x, scope| Ok(ops::tanh(x, scope)?.into()),
        scope,
    )?;

    // Create output layer
    let (vars2, layer2) = layer(
        layer1.clone(),
        hidden_size,
        1,
        &|x, _| Ok(x),
        scope
    )?;

    // Define loss function
    let error = ops::sub(layer2.clone(), label.clone(), scope)?;
    let error_squared = ops::mul(error.clone(), error, scope)?;

    // Setup optimizer
    let mut optimizer = AdadeltaOptimizer::new();
    optimizer.set_learning_rate(ops::constant(1.0f32, scope)?);
    
    let mut variables = Vec::new();
    variables.extend(vars1);
    variables.extend(vars2);

    // Create minimizer operation
    let (_, minimize) = optimizer.minimize(
        scope,
        error_squared.clone().into(),
        MinimizeOptions::default().with_variables(&variables),
    )?;

    // Create TensorFlow session
    let session = Session::new(&SessionOptions::new(), &scope.graph())?;

    // Initialize variables
    let mut run_args = SessionRunArgs::new();
    for var in &variables {
        run_args.add_target(&var.initializer());
    }
    session.run(&mut run_args)?;

    // Training loop
    let mut input_tensor = Tensor::<f32>::new(&[1, 2]);
    let mut label_tensor = Tensor::<f32>::new(&[1]);

    println!("Training XOR model...");
    for i in 0..10000 {
        // Generate XOR training data
        input_tensor[0] = (i & 1) as f32;
        input_tensor[1] = ((i >> 1) & 1) as f32;
        label_tensor[0] = ((i & 1) ^ ((i >> 1) & 1)) as f32;

        // Run training step
        let mut run_args = SessionRunArgs::new();
        run_args.add_target(&minimize);
        run_args.add_feed(&input, 0, &input_tensor);
        run_args.add_feed(&label, 0, &label_tensor);
        session.run(&mut run_args)?;

        if i % 1000 == 0 {
            println!("Step {}", i);
        }
    }

    // Test the model
    println!("\nTesting XOR model:");
    let test_cases = vec![
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
    ];

    for (x1, x2, expected) in test_cases {
        input_tensor[0] = x1;
        input_tensor[1] = x2;
        
        let mut run_args = SessionRunArgs::new();
        let output_fetch = run_args.request_fetch(&layer2, 0);
        run_args.add_feed(&input, 0, &input_tensor);
        session.run(&mut run_args)?;
        
        let output = run_args.fetch::<f32>(output_fetch)?[0];
        println!("Input: ({}, {}), Expected: {}, Got: {:.4}", x1, x2, expected, output);
    }

    Ok(())
}

// Helper function to create a layer
fn layer<O1: Into<Output>>(
    input: O1,
    input_size: u64,
    output_size: u64,
    activation: &dyn Fn(Output, &mut Scope) -> Result<Output, Status>,
    scope: &mut Scope,
) -> Result<(Vec<Variable>, Output), Status> {
    let mut scope = scope.new_sub_scope("layer");
    let scope = &mut scope;
    
    // Create weights
    let w_shape = ops::constant(&[input_size as i64, output_size as i64][..], scope)?;
    let w = Variable::builder()
        .initial_value(
            ops::RandomStandardNormal::new()
                .dtype(DataType::Float)
                .build(w_shape, scope)?,
        )
        .data_type(DataType::Float)
        .shape([input_size, output_size])
        .build(&mut scope.with_op_name("w"))?;

    // Create biases
    let b = Variable::builder()
        .const_initial_value(Tensor::<f32>::new(&[output_size]))
        .build(&mut scope.with_op_name("b"))?;

    // Connect the layer
    Ok((
        vec![w.clone(), b.clone()],
        activation(
            ops::add(
                ops::mat_mul(input, w.output().clone(), scope)?,
                b.output().clone(),
                scope,
            )?
            .into(),
            scope,
        )?,
    ))
}
