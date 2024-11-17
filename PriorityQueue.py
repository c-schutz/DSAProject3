#geeksforgeeks priority queue implementation
class PriorityQueue(object):
    def __init__(self):
        self.queue = []

    def __str__(self):
        return ' '.join([str(i) for i in self.queue])

    def checkEmpty(self):
        return len(self.queue) == 0

    def insert(self, data):
        if isinstance(data, tuple):
            self.queue.append(data)
        else:
            print("Incorrect data type inputted")

    def pop(self):
        try:
            max_val_index = 0
            for i in range(len(self.queue)):
                if self.queue[i][0] > self.queue[max_val_index][0]:
                    max_val_index = i
            item = self.queue[max_val_index]
            del self.queue[max_val_index]
            return item
        except IndexError:
            print("Queue is empty!")
            exit()