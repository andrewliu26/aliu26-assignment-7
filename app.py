from flask import Flask, render_template, request, url_for, session
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Replace with your own secret key, needed for session management


def generate_data(N, mu, beta0, beta1, sigma2, S):
    # Generate data and initial plots

    # Generate random dataset X of size N with values between 0 and 1
    X = np.random.uniform(0, 1, N)

    # Generate random dataset Y using specified parameters
    error = np.random.normal(mu, np.sqrt(sigma2), N)
    Y = beta0 + beta1 * X + error

    # Fit linear regression model
    model = LinearRegression()
    X_reshaped = X.reshape(-1, 1)
    model.fit(X_reshaped, Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    # Generate scatter plot with regression line
    plt.figure(figsize=(10, 6))
    plt.scatter(X, Y, alpha=0.5)
    plt.plot(X, model.predict(X_reshaped), color="red")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Scatter Plot with Fitted Regression Line")
    plt.savefig("static/plot1.png")
    plt.close()

    # Run simulations
    slopes = []
    intercepts = []

    for _ in range(S):
        # Generate simulated datasets
        X_sim = np.random.uniform(0, 1, N)
        error_sim = np.random.normal(mu, np.sqrt(sigma2), N)
        Y_sim = beta0 + beta1 * X_sim + error_sim

        # Fit linear regression to simulated data
        sim_model = LinearRegression()
        X_sim_reshaped = X_sim.reshape(-1, 1)
        sim_model.fit(X_sim_reshaped, Y_sim)
        sim_slope = sim_model.coef_[0]
        sim_intercept = sim_model.intercept_

        slopes.append(sim_slope)
        intercepts.append(sim_intercept)

    # Plot histograms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.hist(slopes, bins=30, alpha=0.7)
    ax1.axvline(slope, color="red", linestyle="dashed")
    ax1.set_title("Distribution of Slopes")
    ax1.set_xlabel("Slope")

    ax2.hist(intercepts, bins=30, alpha=0.7)
    ax2.axvline(intercept, color="red", linestyle="dashed")
    ax2.set_title("Distribution of Intercepts")
    ax2.set_xlabel("Intercept")

    plt.tight_layout()
    plt.savefig("static/plot2.png")
    plt.close()

    # Calculate proportions more extreme than observed
    slope_more_extreme = np.mean(
        np.abs(np.array(slopes) - beta1) >= np.abs(slope - beta1)
    )
    intercept_extreme = np.mean(
        np.abs(np.array(intercepts) - beta0) >= np.abs(intercept - beta0)
    )

    return (
        X,
        Y,
        slope,
        intercept,
        "static/plot1.png",
        "static/plot2.png",
        slope_more_extreme,
        intercept_extreme,
        slopes,
        intercepts,
    )


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input from the form
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        S = int(request.form["S"])

        # Generate data and initial plots
        (
            X,
            Y,
            slope,
            intercept,
            plot1,
            plot2,
            slope_extreme,
            intercept_extreme,
            slopes,
            intercepts,
        ) = generate_data(N, mu, beta0, beta1, sigma2, S)

        # Store data in session
        session["X"] = X.tolist()
        session["Y"] = Y.tolist()
        session["slope"] = slope
        session["intercept"] = intercept
        session["slopes"] = slopes
        session["intercepts"] = intercepts
        session["slope_extreme"] = slope_extreme
        session["intercept_extreme"] = intercept_extreme
        session["N"] = N
        session["mu"] = mu
        session["sigma2"] = sigma2
        session["beta0"] = beta0
        session["beta1"] = beta1
        session["S"] = S

        # Return render_template with variables
        return render_template(
            "index.html",
            plot1=plot1,
            plot2=plot2,
            slope_extreme=slope_extreme,
            intercept_extreme=intercept_extreme,
            N=N,
            mu=mu,
            sigma2=sigma2,
            beta0=beta0,
            beta1=beta1,
            S=S,
        )
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    # This route handles data generation (same as above)
    return index()


@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    # Retrieve data from session
    N = int(session.get("N"))
    S = int(session.get("S"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))

    parameter = request.form.get("parameter")
    test_type = request.form.get("test_type")

    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        simulated_stats = np.array(slopes)
        observed_stat = slope
        hypothesized_value = beta1
    else:
        simulated_stats = np.array(intercepts)
        observed_stat = intercept
        hypothesized_value = beta0

    # Calculate p-value based on test type
    if test_type == "!=":  # two-sided test
        p_value = np.mean(
            np.abs(simulated_stats - hypothesized_value)
            >= np.abs(observed_stat - hypothesized_value)
        )
    elif test_type == ">":  # greater than
        p_value = np.mean(simulated_stats >= observed_stat)
    elif test_type == "<":  # less than
        p_value = np.mean(simulated_stats <= observed_stat)
    else:
        p_value = None

    # Set fun message for very small p-values
    fun_message = (
        "Wow, that's extremely significant! 🎉"
        if p_value and p_value <= 0.0001
        else None
    )

    # Plot histogram with test results
    plt.figure(figsize=(10, 6))
    plt.hist(simulated_stats, bins=30, alpha=0.7)
    plt.axvline(observed_stat, color="red", linestyle="dashed", label="Observed")
    plt.axvline(
        hypothesized_value, color="blue", linestyle="dashed", label="Hypothesized"
    )

    # Add shaded regions based on test type
    if test_type == "!=":
        # Shade both tails for two-sided test
        critical_value = abs(observed_stat - hypothesized_value)
        plt.axvspan(
            hypothesized_value + critical_value,
            max(simulated_stats),
            alpha=0.2,
            color="red",
        )
        plt.axvspan(
            min(simulated_stats),
            hypothesized_value - critical_value,
            alpha=0.2,
            color="red",
        )
    elif test_type == ">":
        # Shade right tail
        plt.axvspan(observed_stat, max(simulated_stats), alpha=0.2, color="red")
    elif test_type == "<":
        # Shade left tail
        plt.axvspan(min(simulated_stats), observed_stat, alpha=0.2, color="red")

    plt.title(f"Distribution of {parameter.capitalize()} with {test_type} Test")
    plt.xlabel(parameter.capitalize())
    plt.legend()
    plt.savefig("static/plot3.png")
    plt.close()

    # Return results to template with test_type included
    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot3="static/plot3.png",
        parameter=parameter,
        test_type=test_type,
        observed_stat=observed_stat,
        hypothesized_value=hypothesized_value,
        N=N,
        beta0=beta0,
        beta1=beta1,
        S=S,
        p_value=p_value,
        fun_message=fun_message,
    )


@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    # Retrieve data from session
    N = int(session.get("N"))
    mu = float(session.get("mu"))
    sigma2 = float(session.get("sigma2"))
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))
    S = int(session.get("S"))
    X = np.array(session.get("X"))
    Y = np.array(session.get("Y"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = np.array(session.get("slopes"))
    intercepts = np.array(session.get("intercepts"))

    parameter = request.form.get("parameter")
    confidence_level = (
        float(request.form.get("confidence_level")) / 100
    )  # Convert to proportion

    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        estimates = slopes
        observed_stat = slope
        true_param = beta1
    elif parameter == "intercept":
        estimates = intercepts
        observed_stat = intercept
        true_param = beta0
    else:
        return "Invalid parameter", 400

    # Calculate mean and standard deviation
    mean_estimate = np.mean(estimates)
    std_estimate = np.std(estimates, ddof=1)  # Use sample standard deviation

    # Calculate confidence interval
    if std_estimate == 0:
        ci_lower = mean_estimate
        ci_upper = mean_estimate
    else:
        # Convert confidence level from percentage to proportion
        t_value = stats.t.ppf((1 + confidence_level) / 2, len(estimates) - 1)
        margin = t_value * std_estimate / np.sqrt(len(estimates))
        ci_lower = mean_estimate - margin
        ci_upper = mean_estimate + margin

    # Check if true parameter is in interval
    includes_true = ci_lower <= true_param <= ci_upper

    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(
        estimates,
        [0] * len(estimates),
        color="gray",
        alpha=0.5,
        label="Simulated Estimates",
    )
    plt.plot([mean_estimate], [0], "bo", label="Mean Estimate")
    plt.hlines(
        0,
        xmin=ci_lower,
        xmax=ci_upper,
        colors="blue",
        linestyle="-",
        linewidth=3,
        label=f"{confidence_level * 100}% Confidence Interval",
    )
    plt.axvline(x=true_param, color="green", linestyle="--", label="True Value")

    plt.title(
        f"{confidence_level * 100}% Confidence Interval for {parameter.capitalize()} (Mean Estimate)"
    )
    plt.xlabel(f"{parameter.capitalize()} Estimate")
    plt.legend()
    plt.tight_layout()
    plt.savefig("static/plot4.png")
    plt.close()

    # Return results to template
    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot4="static/plot4.png",
        parameter=parameter,
        confidence_level=confidence_level * 100,  # Convert back to percentage
        mean_estimate=mean_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        includes_true=includes_true,
        observed_stat=observed_stat,
        N=N,
        mu=mu,
        sigma2=sigma2,
        beta0=beta0,
        beta1=beta1,
        S=S,
    )


if __name__ == "__main__":
    app.run(debug=True)
