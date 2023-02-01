# source https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report
import matplotlib.pyplot as plt
import numpy as np

class PLot_CR():
    def show_values(self, pc, fmt="%.2f", **kw):
        '''
        Heatmap with text in each cell with matplotlib's pyplot
        Source: https://stackoverflow.com/a/25074150/395857 
        By HYRY
        '''
        pc.update_scalarmappable()
        ax = pc.axes
        #ax = pc.axes# FOR LATEST MATPLOTLIB
        #Use zip BELOW IN PYTHON 3
        for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
            x, y = p.vertices[:-2, :].mean(0)
            if np.all(color[:3] > 0.5):
                color = (0.0, 0.0, 0.0)
            else:
                color = (1.0, 1.0, 1.0)
            ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)


    def cm2inch(self, *tupl):
        '''
        Specify figure size in centimeter in matplotlib
        Source: https://stackoverflow.com/a/22787457/395857
        By gns-ank
        '''
        inch = 2.54
        if type(tupl[0]) == tuple:
            return tuple(i/inch for i in tupl[0])
        else:
            return tuple(i/inch for i in tupl)


    def heatmap(self, AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, correct_orientation=False, cmap='RdBu'):
        '''
        Inspired by:
        - https://stackoverflow.com/a/16124677/395857 
        - https://stackoverflow.com/a/25074150/395857
        '''

        # Plot it out
        fig, ax = plt.subplots()    
        #c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdBu', vmin=0.0, vmax=1.0)
        c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap=cmap, vmin=0.0, vmax=1.0)

        # put the major ticks at the middle of each cell
        ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
        ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

        # set tick labels
        #ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
        ax.set_xticklabels(xticklabels, minor=False)
        ax.set_yticklabels(yticklabels, minor=False)

        # set title and x/y labels
        plt.title(title, y=1.25)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)      

        # Remove last blank column
        plt.xlim( (0, AUC.shape[1]) )

        # Turn off all the ticks
        ax = plt.gca()    
        for t in ax.xaxis.get_major_ticks():
            t.tick1line.set_visible(False)
            t.tick2line.set_visible(False)
        for t in ax.yaxis.get_major_ticks():
            t.tick1line.set_visible(False)
            t.tick2line.set_visible(False)

        # Add color bar
        plt.colorbar(c)

        # Add text in each cell 
        self.show_values(c)

        # Proper orientation (origin at the top left instead of bottom left)
        if correct_orientation:
            ax.invert_yaxis()
            ax.xaxis.tick_top()       

        # resize 
        fig = plt.gcf()
        #fig.set_size_inches(cm2inch(40, 20))
        #fig.set_size_inches(cm2inch(40*4, 20*4))
        fig.set_size_inches(self.cm2inch(figure_width, figure_height))



    def plot_classification_report(self, classification_report, classes_names, number_of_classes=7, title='Classification report ', cmap='RdYlGn', save_path="", save=False   ):
        '''
        Plot scikit-learn classification report.
        Extension based on https://stackoverflow.com/a/31689645/395857 
        '''
        lines = classification_report.split('\n')
        #drop initial lines
        lines = lines[2:]

        classes = []
        plotMat = []
        support = []
        class_names = []
        for line in lines[: number_of_classes +4 ]:
            t = list(filter(None, line.strip().split('  ')))
            if len(t) < 4: continue
            classes.append(t[0])
            v = [float(x) for x in t[1: len(t)-1]]
            support.append(int(t[-1]))
            class_names.append(t[0])
            plotMat.append(v)

       
        Acuu = list(filter(None, lines[ number_of_classes +1 ].strip().split('  ')))
        class_names.append(f"OVer All accuracy")
        support.append(int(Acuu[2]))
        plotMat.append([0,0, float(Acuu[1])])
        
        xlabel = 'Metrics'
        ylabel = 'Classes'
        xticklabels = ['Precision', 'Recall', 'F1-score']
        yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
        figure_width = 10
        figure_height = len(class_names) + 3
        correct_orientation = True
        self.heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height, correct_orientation, cmap=cmap)

        if save == True:
            plt.savefig(f"{save_path}/{title}.jpg",bbox_inches ="tight" )

        plt.show()


        
