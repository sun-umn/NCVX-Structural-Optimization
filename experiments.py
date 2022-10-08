# first party
import os

# third party
import matplotlib.pyplot as plt
import pandas as pd


def build_structure_comparison_data(name: str) -> pd.DataFrame:
    """
    The goal of this function is to gather results from different
    problems by name for comparison.

    NOTE: At the time of function creation 10/82022 the methods are:
    1. cnn-lbfgs (Google)
    2. cnn-pygranso (UMN)
    3. cnn-adam (UMN)
    """
    # default prefix list
    experiment_prefix_list = ["google", "pygranso"]

    # Set the path
    path = "./results"

    # Check that the files exist
    for experiment in experiment_prefix_list:
        filename = f"{experiment}_{name}_results.csv"
        if not os.path.exists(os.path.join(path, filename)):
            raise ValueError(
                "Need to gather results for " f"experiment {experiment}_{name}"
            )

    # Load results from google
    google_filename = f"google_{name}_results.csv"
    google_results = pd.read_csv(os.path.join(path, google_filename))  # noqa

    # Set the index to the steps
    # First check that step is a column
    if "step" not in google_results.columns:
        raise ValueError("Google - expected column named step!")

    # Set the index
    google_results = google_results.set_index("step")

    # Load the pygranso files
    pygranso_filename = f"pygranso_{name}_results.csv"
    pygranso_results = pd.read_csv(
        os.path.join(path, pygranso_filename)
    )  # noqa

    # Similarly set the index to step
    if "step" not in pygranso_results.columns:
        raise ValueError("Pygranso - expected column named step!")

    # Set the index
    pygranso_results = pygranso_results.set_index("step")

    # Merge the data frames together
    # If one has less iterations just forward fill the results
    results = pd.merge(
        pygranso_results, google_results, on="step", how="left"
    ).ffill()  # noqa

    return results


def save_loss_plots(name: str) -> None:
    """
    Function to gather the results of the data and create plots to
    save for our papers regarding structural optimization
    """
    path = "./images"
    results = build_structure_comparison_data(name=name)

    # Create the figure & axis
    fig, ax = plt.subplots(1, 1)

    # plot the results
    results.plot(lw=2, ax=ax)

    # Set attributes of the plot
    title = name.lower().split("_")
    title = list(map(lambda x: x.capitalize(), title))
    title = " ".join(title)
    title = f"{title}: Method Comparisons"
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel("Compliance (loss function)")
    ax.grid()

    # Save the figure
    plt.savefig(os.path.join(path, f"{name}_fig.png"))
