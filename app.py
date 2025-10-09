from flask import Flask, render_template, request, jsonify
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

app = Flask(__name__)

# --------------------
# Fonctions Black-Scholes
# --------------------
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    C = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)
    return C, delta, gamma, vega, theta

# --------------------
# Arbre binomial
# --------------------
def binomial_tree_call(S, K, T, r, sigma, N):
    dt = T/N
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    p = (np.exp(r*dt) - d)/(u - d)

    # Génération des prix
    ST = np.zeros((N+1, N+1))
    for i in range(N+1):
        for j in range(i+1):
            ST[j,i] = S * (u**(i-j)) * (d**j)

    # Calcul des payoffs
    C = np.maximum(ST[:,N] - K, 0)

    # Backward induction
    for i in range(N-1, -1, -1):
        for j in range(i+1):
            C[j] = np.exp(-r*dt)*(p*C[j] + (1-p)*C[j+1])

    return C[0], ST

# --------------------
# Arbre trinomial
# --------------------
def trinomial_tree_call(S, K, T, r, sigma, N):
    dt = T / N
    u = np.exp(sigma*np.sqrt(2*dt))
    d = 1/u
    m = 1
    R = np.exp(r*dt)
    pu = ((np.sqrt(R)-np.sqrt(d))/(np.sqrt(u)-np.sqrt(d)))**2
    pd = ((np.sqrt(u)-np.sqrt(R))/(np.sqrt(u)-np.sqrt(d)))**2
    pm = 1 - pu - pd

    # Génération des prix
    size = 2*N + 1
    ST = np.zeros((size,N+1))
    ST[N,0] = S
    for i in range(1,N+1):
        for j in range(N-i, N+i+1):
            ST[j,i] = ST[j,i-1]*u if j > N else (ST[j,i-1]*d if j < N else ST[j,i-1]*1)

    # Payoffs à maturité
    payoffs = np.maximum(ST[:,N] - K,0)

    return np.max(payoffs), ST

# --------------------
# Génération graphe interactif arbre
# --------------------
def plot_tree(ST, K):
    N = ST.shape[1]-1
    fig = go.Figure()

    for i in range(N+1):
        for j in range(i+1):
            fig.add_trace(go.Scatter(
                x=[i,i+1],
                y=[ST[j,i], ST[j,i+1] if i+1<N+1 and j<i+1 else ST[j,i]],
                mode='lines+markers+text',
                text=[f'{ST[j,i]:.2f}', f'{ST[j,i+1]:.2f}' if i+1<N+1 and j<i+1 else ''],
                textposition="top center"
            ))
    fig.update_layout(title='Arbre des prix du sous-jacent', xaxis_title='Période', yaxis_title='Prix')
    return fig.to_json()

# --------------------
# Route principale
# --------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    result = {}
    if request.method == 'POST':
        S = float(request.form['S'])
        K = float(request.form['K'])
        T = float(request.form['T'])
        r = float(request.form['r'])
        sigma = float(request.form['sigma'])
        model = request.form['model']
        N = int(request.form.get('N', 10))

        if model == 'Black-Scholes':
            price, delta, gamma, vega, theta = black_scholes_call(S,K,T,r,sigma)
            result = {
                'price': round(price,2),
                'delta': round(delta,2),
                'gamma': round(gamma,2),
                'vega': round(vega,2),
                'theta': round(theta,2),
                'tree': None
            }
        elif model == 'Binomial':
            price, ST = binomial_tree_call(S,K,T,r,sigma,N)
            tree_json = plot_tree(ST,K)
            result = {'price': round(price,2), 'tree': tree_json}
        elif model == 'Trinomial':
            price, ST = trinomial_tree_call(S,K,T,r,sigma,N)
            tree_json = plot_tree(ST,K)
            result = {'price': round(price,2), 'tree': tree_json}

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

