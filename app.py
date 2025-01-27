import dash
from dash import html, dcc, Input, Output, State
import plotly.graph_objects as go
import torch
from models import get_model, SingleNeuronModel, MultiLayerModel
from data import get_dataset
from training import Trainer
import numpy as np

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        dcc.Graph(id='main-plot'),
        dcc.Graph(id='loss-plot')
    ], style={'width': '60%', 'display': 'inline-block'}),
    
    html.Div([
        html.H4('Controls'),
        html.Label('Dataset Type:'),
        dcc.Dropdown(
            id='dataset-type',
            options=[
                {'label': 'Linear', 'value': 'linear'},
                {'label': 'Non-linear', 'value': 'nonlinear'},
                {'label': 'Classification', 'value': 'classification'}
            ],
            value='linear'
        ),
        
        html.Label('Model Type:'),
        dcc.Dropdown(
            id='model-type',
            options=[
                {'label': 'Single Neuron', 'value': 'single_neuron'},
                {'label': 'Multi-layer Network', 'value': 'multi_layer'},
                {'label': 'Deep Network (20 layers)', 'value': 'deep_multi_layer'},
                {'label': 'Deep Network + BatchNorm', 'value': 'deep_batchnorm'},
                {'label': 'Binary Classifier', 'value': 'binary_classifier'}
            ],
            value='single_neuron'
        ),
        
        html.Label('Batch Size:'),
        dcc.Dropdown(
            id='batch-size',
            options=[
                {'label': '2 samples', 'value': 2},
                {'label': '10 samples', 'value': 10},
                {'label': '30 samples', 'value': 30},
                {'label': '100 samples', 'value': 100},
                {'label': '500 samples', 'value': 500},
                {'label': '2000 samples', 'value': 2000}
            ],
            value=2,
            style={'marginBottom': '20px'}
        ),
        
        html.Label('Number of Data Points:'),
        dcc.Dropdown(
            id='n-points',
            options=[
                {'label': '30 points', 'value': 30},
                {'label': '100 points', 'value': 100},
                {'label': '500 points', 'value': 500},
                {'label': '2000 points', 'value': 2000}
            ],
            value=100,
            style={'marginBottom': '20px'}
        ),
        
        html.Label('Learning Rate:'),
        dcc.Dropdown(
            id='learning-rate',
            options=[
                {'label': '1.0', 'value': 1.0},
                {'label': '0.1', 'value': 0.1},
                {'label': '0.01', 'value': 0.01},
                {'label': '0.001', 'value': 0.001}
            ],
            value=1.0,
            style={'marginBottom': '20px'}
        ),
        
        dcc.Checklist(
            id='show-backprop',
            options=[{'label': 'Show Backpropagation', 'value': 'show'}],
            value=[]
        ),
        
        html.Button('Step', id='step-button'),
        html.Button('Step 500', id='step-500-button'),
        html.Button('Reset', id='reset-button'),
        
        html.Div([
            html.H4('Backpropagation Details', 
                   style={'marginTop': '20px', 'marginBottom': '10px'}),
            html.Div(id='backprop-viz')
        ], style={'marginTop': '20px'}),
        
        html.Div([
            html.H4('Model Parameters'),
            html.Pre(id='weights-display', 
                    style={
                        'border': '1px solid #ddd',
                        'padding': '10px',
                        'backgroundColor': '#f8f9fa',
                        'fontFamily': 'monospace',
                        'whiteSpace': 'pre-wrap',
                        'fontSize': '12px'
                    })
        ])
    ], style={'width': '35%', 'float': 'right', 'padding': '20px'})
])

# Initialize global state
current_data = None
current_model = None
current_trainer = None

@app.callback(
    [Output('main-plot', 'figure'),
     Output('loss-plot', 'figure'),
     Output('backprop-viz', 'children'),
     Output('weights-display', 'children')],
    [Input('dataset-type', 'value'),
     Input('model-type', 'value'),
     Input('step-button', 'n_clicks'),
     Input('step-500-button', 'n_clicks'),
     Input('reset-button', 'n_clicks'),
     Input('show-backprop', 'value'),
     Input('learning-rate', 'value'),
     Input('n-points', 'value')],
    [State('batch-size', 'value')]
)
def update_plot(dataset_type, model_type, step_clicks, step_500_clicks, 
                reset_clicks, show_backprop, learning_rate, n_points, batch_size):
    global current_data, current_model, current_trainer
    
    ctx = dash.callback_context
    if not ctx.triggered:
        trigger_id = 'initial'
    else:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id in ['dataset-type', 'model-type', 'n-points'] or current_data is None:
        # Set seeds for reproducibility
        torch.manual_seed(2)
        np.random.seed(2)
        
        train_data, val_data, test_data = get_dataset(dataset_type, n_points=n_points)
        current_data = (train_data, val_data, test_data)
        current_model = get_model(model_type)
        current_trainer = Trainer(current_model, learning_rate=learning_rate)
    elif trigger_id == 'learning-rate':
        # Only update the learning rate without resetting the model
        for param_group in current_trainer.optimizer.param_groups:
            param_group['lr'] = learning_rate
    
    # Add reset button logic
    if trigger_id == 'reset-button' and reset_clicks and reset_clicks > 0:
        # Set seeds for reproducibility
        torch.manual_seed(2)
        np.random.seed(2)
        
        # Keep the same data but reset model and trainer
        current_model = get_model(model_type)
        current_trainer = Trainer(current_model, learning_rate=learning_rate)
    
    train_data, val_data, test_data = current_data
    
    # Handle step-500 button click
    if trigger_id == 'step-500-button' and step_500_clicks and step_500_clicks > 0:
        print("Performing 500 steps...")
        for _ in range(500):
            x, y = train_data
            indices = torch.randperm(len(x))[:batch_size]
            batch_x = x[indices]
            batch_y = y[indices]
            _, _ = current_trainer.training_step(batch_x, batch_y)
            
            # Compute full training and validation losses
            train_loss = current_trainer.compute_loss(train_data[0], train_data[1])
            val_loss = current_trainer.validate(val_data[0], val_data[1])
            current_trainer.train_losses.append(train_loss)
            current_trainer.val_losses.append(val_loss)
            
            if len(current_trainer.train_losses) % 100 == 0:
                print(f"Step {len(current_trainer.train_losses)}, "
                      f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
    
    # Handle step button click
    if trigger_id == 'step-button' and step_clicks and step_clicks > 0:
        print("Step button clicked")
        x, y = train_data
        indices = torch.randperm(len(x))[:batch_size]
        batch_x = x[indices]
        batch_y = y[indices]
        _, gradients = current_trainer.training_step(batch_x, batch_y)
        
        # Compute full training and validation losses
        train_loss = current_trainer.compute_loss(train_data[0], train_data[1])
        val_loss = current_trainer.validate(val_data[0], val_data[1])
        current_trainer.train_losses.append(train_loss)
        current_trainer.val_losses.append(val_loss)
    
    # Create main plot
    fig = go.Figure()
    
    # Plot training data points
    train_x, train_y = train_data
    fig.add_trace(go.Scatter(
        x=train_x.numpy().flatten(),
        y=train_y.numpy().flatten(),
        mode='markers',
        name='Training Data',
        marker=dict(color='blue', size=8)
    ))
    
    # Plot validation data points
    val_x, val_y = val_data
    fig.add_trace(go.Scatter(
        x=val_x.numpy().flatten(),
        y=val_y.numpy().flatten(),
        mode='markers',
        name='Validation Data',
        marker=dict(color='green', size=8)
    ))
    
    # Plot test data points
    test_x, test_y = test_data
    fig.add_trace(go.Scatter(
        x=test_x.numpy().flatten(),
        y=test_y.numpy().flatten(),
        mode='markers',
        name='Test Data',
        marker=dict(color='red', size=8)
    ))
    
    # Highlight batch points if step button was clicked
    if trigger_id == 'step-button' and step_clicks and step_clicks > 0:
        fig.add_trace(go.Scatter(
            x=batch_x.numpy().flatten(),
            y=batch_y.numpy().flatten(),
            mode='markers',
            name='Current Batch',
            marker=dict(color='orange', size=12, symbol='circle-open', line_width=2)
        ))
    
    # Plot model prediction
    with torch.no_grad():
        x_plot = torch.linspace(-5, 5, 100).reshape(-1, 1)
        y_pred = current_model(x_plot)
        fig.add_trace(go.Scatter(
            x=x_plot.numpy().flatten(),
            y=y_pred.numpy().flatten(),
            mode='lines',
            name='Model Prediction',
            line=dict(color='orange', width=4)
        ))
    
    # Update main plot layout
    fig.update_layout(
        title='Basics of Artificial Neural Networks',
        xaxis_title='X',
        yaxis_title='Y',
        showlegend=True,
        plot_bgcolor='white',
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    # Create loss plot
    loss_fig = go.Figure()
    
    if len(current_trainer.train_losses) > 0:
        # Plot training loss
        loss_fig.add_trace(go.Scatter(
            x=list(range(len(current_trainer.train_losses))),
            y=current_trainer.train_losses,
            mode='lines+markers',
            name='Training Loss',
            line=dict(color='blue', width=2)
        ))
        
        # Plot validation loss
        loss_fig.add_trace(go.Scatter(
            x=list(range(len(current_trainer.val_losses))),
            y=current_trainer.val_losses,
            mode='lines+markers',
            name='Validation Loss',
            line=dict(color='red', width=2)
        ))
        
        # Update loss plot layout
        loss_fig.update_layout(
            title='Training Progress',
            xaxis_title='Steps',
            yaxis_title='Loss (MSE)',
            showlegend=True,
            plot_bgcolor='white',
            margin=dict(l=40, r=40, t=40, b=40),
            yaxis_type='log'  # Use log scale for loss
        )
    else:
        # Initial empty plot
        loss_fig.update_layout(
            title='Training Progress',
            xaxis_title='Steps',
            yaxis_title='Loss (MSE)',
            showlegend=True,
            plot_bgcolor='white',
            margin=dict(l=40, r=40, t=40, b=40),
            yaxis_type='log'
        )
    
    backprop_viz = html.Div("Enable 'Show Backpropagation' and click 'Step' to see the computation details.", 
                           style={'color': '#666', 'fontStyle': 'italic'})
    
    if show_backprop and 'show' in show_backprop and trigger_id == 'step-button':
        # Get the current batch computations
        with torch.autograd.grad_mode.enable_grad():
            # Forward pass computations
            if isinstance(current_model, SingleNeuronModel):
                w = current_model.linear.weight.item()
                b = current_model.linear.bias.item()
                
                # Forward pass - ensure tensors require gradients
                batch_x_grad = batch_x.clone().detach().requires_grad_(True)
                z = batch_x_grad * w + b
                pred = z
                loss = ((pred - batch_y) ** 2).mean()
                
                # Backward pass (compute gradients)
                loss.backward()
                
                # Get gradients
                dw = current_model.linear.weight.grad.item()
                db = current_model.linear.bias.grad.item()
                
                # Create visualization text
                backprop_viz = html.Pre(
                    f"""Forward Pass:
    Input (x): {batch_x.numpy().flatten()}
    
    Linear Layer:
        w = {w:.4f}, b = {b:.4f}
        z = w * x + b = {z.detach().numpy().flatten()}
    
    Prediction (ŷ): {pred.detach().numpy().flatten()}
    
    Target (y): {batch_y.numpy().flatten()}
    
    Loss (MSE): {loss.item():.4f}
        MSE = mean((ŷ - y)²)
    
Backward Pass:
    ∂Loss/∂pred = 2(ŷ - y)/n = {2*(pred.detach() - batch_y).mean().item():.4f}
    ∂Loss/∂w = {dw:.4f}
    ∂Loss/∂b = {db:.4f}""",
                    style={
                        'backgroundColor': '#f8f9fa',
                        'padding': '10px',
                        'border': '1px solid #ddd',
                        'borderRadius': '5px',
                        'fontFamily': 'monospace',
                        'whiteSpace': 'pre-wrap'
                    }
                )
            else:  # MultiLayerModel
                # Forward pass computations with gradient tracking
                x = batch_x.clone().detach().requires_grad_(True)
                layer1_w = current_model.network[0].weight
                layer1_b = current_model.network[0].bias
                z1 = x @ layer1_w.t() + layer1_b
                a1 = torch.relu(z1)
                
                layer2_w = current_model.network[2].weight
                layer2_b = current_model.network[2].bias
                z2 = a1 @ layer2_w.t() + layer2_b
                a2 = torch.relu(z2)
                
                layer3_w = current_model.network[4].weight
                layer3_b = current_model.network[4].bias
                z3 = a2 @ layer3_w.t() + layer3_b
                pred = z3
                
                loss = ((pred - batch_y) ** 2).mean()
                loss.backward()
                
                backprop_viz = html.Pre(
                    f"""Forward Pass:
    Input (x): {batch_x.numpy().flatten()}
    
    Layer 1 (Linear + ReLU):
        z1 = w1 * x + b1 = {z1.detach().numpy().flatten()}
        a1 = ReLU(z1) = {a1.detach().numpy().flatten()}
    
    Layer 2 (Linear + ReLU):
        z2 = w2 * a1 + b2 = {z2.detach().numpy().flatten()}
        a2 = ReLU(z2) = {a2.detach().numpy().flatten()}
    
    Layer 3 (Linear):
        z3 = w3 * a2 + b3 = {z3.detach().numpy().flatten()}
    
    Prediction (ŷ): {pred.detach().numpy().flatten()}
    Target (y): {batch_y.numpy().flatten()}
    
    Loss (MSE): {loss.item():.4f}
        MSE = mean((ŷ - y)²)
    
Backward Pass:
    ∂Loss/∂pred = 2(ŷ - y)/n = {2*(pred.detach() - batch_y).mean().item():.4f}
    
    Layer 3 gradients:
        ∂Loss/∂w3: {current_model.network[4].weight.grad.numpy().flatten()}
        ∂Loss/∂b3: {current_model.network[4].bias.grad.numpy().flatten()}
    
    Layer 2 gradients:
        ∂Loss/∂w2: {current_model.network[2].weight.grad.numpy().flatten()}
        ∂Loss/∂b2: {current_model.network[2].bias.grad.numpy().flatten()}
    
    Layer 1 gradients:
        ∂Loss/∂w1: {current_model.network[0].weight.grad.numpy().flatten()}
        ∂Loss/∂b1: {current_model.network[0].bias.grad.numpy().flatten()}""",
                    style={
                        'backgroundColor': '#f8f9fa',
                        'padding': '10px',
                        'border': '1px solid #ddd',
                        'borderRadius': '5px',
                        'fontFamily': 'monospace',
                        'whiteSpace': 'pre-wrap'
                    }
                )

    # Create weights display string
    weights_str = "Model Parameters:\n"
    for name, param in current_model.named_parameters():
        weights_str += f"\n{name}:\n{param.data.numpy()}\n"
    
    return fig, loss_fig, backprop_viz, weights_str

if __name__ == '__main__':
    app.run_server(debug=True) 