import visdom

vis = visdom.Visdom(env='main',port=8098)
is_loss_line_initialized = False
is_classification_accuracy_line_initialized = False
is_output_text_initialized = False
win_line_loss = None
win_line_classification_accuracy = None


def loss_plot(x, y, name):
    global win_line_loss
    global is_loss_line_initialized

    if not is_loss_line_initialized:
        win_line_loss = vis.line(X=x, Y=y, win='loss', name=name, update=None)
        is_loss_line_initialized = True
    else:
        vis.line(X=x, Y=y, win=win_line_loss, name=name, update='append')

def classification_accuracy_plot(x, y, name):
    global win_line_classification_accuracy
    global is_classification_accuracy_line_initialized

    if not is_classification_accuracy_line_initialized:
        win_line_classification_accuracy = vis.line(X=x, Y=y, win='classification_accuracy', name=name, update=None)
        is_classification_accuracy_line_initialized = True
    else:
        vis.line(X=x, Y=y, win='classification_accuracy', name=name, update='append')


