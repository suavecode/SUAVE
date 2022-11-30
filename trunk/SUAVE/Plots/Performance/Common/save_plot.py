## @ingroup Plots-Performance-Common
# save_plot.py
# 
# Created:    Nov 2022, J. Smart
# Modified:   


## @ingroup Plots-Performance-Common
def save_plot(fig, save_filename, file_type, *args, **kwargs):
    """Save a plot, with an import check for kaleido

    Assumptions:
    None

    Source:
    None

    Inputs:
    fig             Plotly graph object

    Outputs: 
    N/A

    Properties Used:
    N/A	
    """

    try:
        import kaleido
        fig.write_image(save_filename.replace("_", " ") + file_type)
    except ImportError:
        raise ImportError(
            'You need to install kaleido to save the figure.')

    return
