import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_cars(ax,cars):
    for car in cars:
        fcolor = 'g'
        if car.id == 0:
            fcolor = 'r'
        rect = patches.Polygon(car.vertices[:4],linewidth=1, edgecolor='black', facecolor=fcolor)


        # Add the patch to the axis
        ax.add_patch(rect)