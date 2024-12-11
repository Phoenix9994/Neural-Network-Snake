import matplotlib.pyplot as plt
from IPython import display
import pygame
plt.style.use('dark_background')
plt.ion()

pygame.mixer.init()

pygame.mixer.music.load('music3.mp3')
pygame.mixer.music.set_volume(0.5) 
pygame.mixer.music.play(-1)  

#Make a graph plot, similar to r
def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...', color='white', fontweight='bold') 
    plt.xlabel('Number of Games', color='white', fontweight='bold') 
    plt.ylabel('Score', color='white', fontweight='bold') 

    # Plot with bolder lines
    plt.plot(scores, color='cyan', linewidth=4, label='Scores')
    plt.plot(mean_scores, color='white', linewidth=4, label='Mean Scores')  

    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)