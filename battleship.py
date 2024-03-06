import tkinter as tk
from tkinter import ttk
import copy
import random

class Agent:
    def __init__(self, size):
        self.size = size
        self.moves = set()
        self.last_hit = None  
        self.attack=False

    def valid_moves(self, board):
        return [(row, col) for row in range(self.size) for col in range(self.size) if (row, col) not in self.moves]

    def make_move(self, board, alive_ships,first_unsunk_hit):
        row, col = self.choose_move(board, alive_ships,first_unsunk_hit)
        self.moves.add((row, col))
        return row, col

    def choose_move(self, board, alive_ships,first_unsunk_hit):
        raise NotImplementedError("This method should be overridden")


class RandomAgent(Agent):
    def choose_move(self, board, alive_ships,first_unsunk_hit):
        #print(self.valid_moves(board))
        return random.choice(self.valid_moves(board))


class HitNearbyAgent(Agent):
    def choose_move(self, board, alive_ships,first_unsunk_hit):
        if self.last_hit is not None:
            # If the last move hit a ship, try to hit nearby cells
            row, col = self.last_hit
            #print(row, col)
            nearby_cells = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
            valid_nearby_cells = [cell for cell in nearby_cells if cell in self.valid_moves(board)]
            if valid_nearby_cells:
                return random.choice(valid_nearby_cells)

        # If the last move did not hit a ship or there are no valid nearby cells, choose a random move
        return random.choice(self.valid_moves(board))

class HitAllNearbyAgent(Agent):
    def __init__(self, size):
        super().__init__(size)
        self.first_hit = None
        self.pending_moves = []
        self.first_pending_moves = []
        

    def choose_move(self, board, alive_ships,first_unsunk_hit):
        #print(alive_ships)
        if self.last_hit is not None:
            # If the last move hit a ship, try to hit nearby cells
            row, col = self.last_hit
            #print("success at",row,col)
            #print(row, col)
            nearby_cells = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
            valid_nearby_cells = [cell for cell in nearby_cells if cell in self.valid_moves(board)]
            self.pending_moves = valid_nearby_cells
            if self.first_hit is None:
                self.first_hit = (row, col)
                self.first_pending_moves = nearby_cells
            #print(self.pending_moves)

        if self.pending_moves:
            move = random.choice(self.pending_moves)
            self.pending_moves.remove(move)
            return move
        
        if self.first_pending_moves:
            #print("good")
            self.first_pending_moves = [cell for cell in self.first_pending_moves if cell in self.valid_moves(board)]
            #print("first",self.first_pending_moves)
            if len(self.first_pending_moves)>0:
                move = random.choice(self.first_pending_moves)
                self.first_pending_moves.remove(move)
                return move
        
        self.first_hit = None
        # If the last move did not hit a ship or there are no valid nearby cells, choose a random move
        return random.choice(self.valid_moves(board))

class DirectionalHitAgent(Agent):
    def __init__(self, size):
        super().__init__(size)
        self.first_hit = None
        self.pending_moves = []
        self.first_pending_moves = []
        self.direction = None

    def choose_move(self, board, alive_ships,first_unsunk_hit):
        #print(alive_ships)
        #print(self.first_hit,self.direction,self.pending_moves)
        if self.last_hit is not None:
            # If the last move hit a ship, try to hit nearby cells
            row, col = self.last_hit
            #print("success at",row,col)
            #print(row, col)
            if self.first_hit is None:
                self.first_hit = (row, col)
                self.first_pending_moves = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
                
            elif self.direction is None:
                self.direction = "horizontal" if row == self.first_hit[0] else "vertical"
                
            if self.direction is not None:
                if self.direction == "horizontal":
                    nearby_cells = [(row, col-1), (row, col+1)]
                else:
                    nearby_cells = [(row-1, col), (row+1, col)]
                valid_nearby_cells = [cell for cell in nearby_cells if cell in self.valid_moves(board)]
                self.pending_moves = valid_nearby_cells
            else:
                nearby_cells = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
                valid_nearby_cells = [cell for cell in nearby_cells if cell in self.valid_moves(board)]
                self.pending_moves = valid_nearby_cells
            
                
            #print(self.pending_moves)

        if self.pending_moves:
            move = random.choice(self.pending_moves)
            self.pending_moves.remove(move)
            return move
        
        if self.first_pending_moves:
            #print("good")
            if self.direction is not None:
                if self.direction == "horizontal":
                    self.first_pending_moves = [cell for cell in self.first_pending_moves if cell in self.valid_moves(board) and cell[0] == self.first_hit[0]]
                else:
                    self.first_pending_moves = [cell for cell in self.first_pending_moves if cell in self.valid_moves(board) and cell[1] == self.first_hit[1]]    
            else:
                self.first_pending_moves = [cell for cell in self.first_pending_moves if cell in self.valid_moves(board)]
            #print("first",self.first_pending_moves)
            if len(self.first_pending_moves)>0:
                move = random.choice(self.first_pending_moves)
                self.first_pending_moves.remove(move)
                return move
        
        self.first_hit = None
        self.direction = None
        # If the last move did not hit a ship or there are no valid nearby cells, choose a random move
        return random.choice(self.valid_moves(board))

class DirectionalSunkAgent(Agent):
    def __init__(self, size):
        super().__init__(size)
        self.first_hit = None
        self.pending_moves = []
        self.first_pending_moves = []
        self.direction = None
        self.last_alive_ships = None 
        self.last_chance = False
        self.x_coord = None
        self.y_coord = None
        
    def choose_move(self, board, alive_ships,first_unsunk_hit):
        """
        for b in board:
            print(b)
        """
        #print(self.last_chance,alive_ships)
        
        if self.last_alive_ships is not None and self.last_alive_ships != alive_ships:
            #print("A ship has sunk!")
            if len(first_unsunk_hit) > 0 :
                for i in first_unsunk_hit:
                    row,col = i[0],i[1]
                    nearby_cells = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
                    valid_nearby_cells = [cell for cell in nearby_cells if cell in self.valid_moves(board)]
                    if len(valid_nearby_cells) > 0:
                        x_coord,y_coord = row,col   
                        break    
                    else:
                        x_coord,y_coord = None,None
                        
                if x_coord is not None and y_coord is not None: 

                    #print(f"Coordinates of an unsunk hit: ({x_coord}, {y_coord})")
                    self.first_hit = None
                    self.direction = None
                    self.last_hit = (x_coord, y_coord)
                
        elif self.last_chance == True:
            #print('m')
            self.last_chance = False
            x_coord, y_coord = self.x_coord, self.y_coord
            assert x_coord is not None and y_coord is not None
            self.first_hit = None
            self.direction = None
            self.last_hit = (x_coord, y_coord)
        #print(self.last_chance)    
        self.last_alive_ships = alive_ships
        #print(alive_ships)
        #print(self.last_hit,self.first_hit,self.direction,self.pending_moves)
        if self.last_hit is not None:
            # If the last move hit a ship, try to hit nearby cells
            row, col = self.last_hit
            #print("success at",row,col)
            #print(row, col)
            if self.first_hit is None:
                self.first_hit = (row, col)
                self.first_pending_moves = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
                
            elif self.direction is None:
                self.direction = "horizontal" if row == self.first_hit[0] else "vertical"
                
            if self.direction is not None:
                if self.direction == "horizontal":
                    nearby_cells = [(row, col-1), (row, col+1)]
                else:
                    nearby_cells = [(row-1, col), (row+1, col)]
                valid_nearby_cells = [cell for cell in nearby_cells if cell in self.valid_moves(board)]
                self.pending_moves = valid_nearby_cells
            else:
                nearby_cells = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
                valid_nearby_cells = [cell for cell in nearby_cells if cell in self.valid_moves(board)]
                self.pending_moves = valid_nearby_cells
        #print(self.last_hit,self.first_hit,self.direction,self.pending_moves,self.first_pending_moves)
                
            #print(self.pending_moves)

        if self.pending_moves:
            move = random.choice(self.pending_moves)
            self.pending_moves.remove(move)
            return move
        
        if self.first_pending_moves:
            #print("good")
            if self.direction is not None:
                if self.direction == "horizontal":
                    self.first_pending_moves = [cell for cell in self.first_pending_moves if cell in self.valid_moves(board) and cell[0] == self.first_hit[0]]
                else:
                    self.first_pending_moves = [cell for cell in self.first_pending_moves if cell in self.valid_moves(board) and cell[1] == self.first_hit[1]]    
            else:
                #print("oui")
                self.first_pending_moves = [cell for cell in self.first_pending_moves if cell in self.valid_moves(board)]
                #print(self.first_pending_moves)
            #print("first",self.first_pending_moves)
            if len(self.first_pending_moves)>0:
                move = random.choice(self.first_pending_moves)
                self.first_pending_moves.remove(move)
                #print("VERY good")
                return move
        
        if len(first_unsunk_hit) > 0 :
                for i in first_unsunk_hit:
                    row,col = i[0],i[1]
                    nearby_cells = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
                    valid_nearby_cells = [cell for cell in nearby_cells if cell in self.valid_moves(board)]
                    if len(valid_nearby_cells) > 0:
                        x_coord,y_coord = row,col   
                        break     
                    else:
                        x_coord,y_coord = None,None
                if x_coord is not None and y_coord is not None:
                    
                    self.x_coord = x_coord
                    self.y_coord = y_coord
                    self.last_chance = True
                    return self.choose_move(board, alive_ships,first_unsunk_hit)
        else:
            self.first_hit = None
            self.direction = None
        # If the last move did not hit a ship or there are no valid nearby cells, choose a random move
        return random.choice(self.valid_moves(board))
    

class Ship:
    def __init__(self, size, row, col, direction):
        self.size = size
        self.row = row
        self.col = col
        self.direction = direction
        self.hits = 0

    def is_sunk(self):
        return self.hits == self.size


class BoardGUI:
    
    def auto_play(self):
        a=0
        while True:
            if all(all(cell != 'O' for cell in row) for row in self.boards[-1]):
                if self.display:
                    self.text_area.insert(tk.END, "\nAll ships sunk! Game over.\n")
                self.game_over = True
                #print(a,"a")
                break
            
            a+=1
            self.next_step()
            #print(self.current_step)
            #print(self.boards[-1])
            
        #print('yo')
        
    def __init__(self, master, agent, size=10, ships=[5,4,3,3,2], display=True):
        self.master = master
        self.size = size
        self.boards = []
        self.moves = []
        self.current_step = 0
        self.score = 0
        self.game_over = False
        self.agent = agent
        self.display = display
        self.ships = [Ship(size, 0, 0, "horizontal") for size in ships]

        if self.display:
            self.frame = tk.Frame(self.master)
            self.frame.pack()
            self.create_widgets()
        self.place_ships()
        self.auto_play()

    def create_widgets(self):
        self.text_area = tk.Text(self.frame, width=60, height=30)
        self.text_area.pack()

        self.moves_label = tk.Label(self.frame, text="Score: 0")
        self.moves_label.pack()

        self.slider = ttk.Scale(self.frame, from_=0, to=0, orient=tk.HORIZONTAL, command=self.on_scale_move)
        self.slider.pack(fill=tk.X)
        
        self.button_frame = tk.Frame(self.frame)
        self.button_frame.pack()
        self.prev_button = tk.Button(self.button_frame, text="Previous Step", command=self.prev_step_button)
        self.prev_button.grid(row=0, column=0)
        self.next_button = tk.Button(self.button_frame, text="Next Step", command=self.next_step_button)
        self.next_button.grid(row=0, column=1)

    def place_ships(self):
        board = [[' ' for _ in range(self.size)] for _ in range(self.size)]
        for ship in self.ships:
            while True:
                row = random.randint(0, self.size - 1)
                col = random.randint(0, self.size - 1)
                direction = random.choice(["vertical", "horizontal"])
                if direction == "horizontal" and col + ship.size <= self.size and all(board[row][col + i] == ' ' for i in range(ship.size)):
                    for i in range(ship.size):
                        board[row][col + i] = 'O'
                    ship.row, ship.col, ship.direction = row, col, direction
                    break
                elif direction == "vertical" and row + ship.size <= self.size and all(board[row + i][col] == ' ' for i in range(ship.size)):
                    for i in range(ship.size):
                        board[row + i][col] = 'O'
                    ship.row, ship.col, ship.direction = row, col, direction
                    break
        self.boards.append(board)
        self.update_board()
        
    def get_alive_ships(self):
        return [ship.size for ship in self.ships if not ship.is_sunk()]
    
    def get_first_unsunk_hit(self):
        for i, row in enumerate(self.boards[self.current_step]):
            for j, cell in enumerate(row):
                if cell == 'X':
                    for ship in self.ships:
                        if ship.direction == "horizontal" and ship.row == i and ship.col <= j < ship.col + ship.size and not ship.is_sunk():
                            return i, j
                        elif ship.direction == "vertical" and ship.col == j and ship.row <= i < ship.row + ship.size and not ship.is_sunk():
                            return i, j
        return None, None
    
    def get_unsunk_hits(self):
        unsunk_hits = []
        for i, row in enumerate(self.boards[self.current_step]):
            for j, cell in enumerate(row):
                if cell == 'X':
                    for ship in self.ships:
                        if ship.direction == "horizontal" and ship.row == i and ship.col <= j < ship.col + ship.size and not ship.is_sunk():
                            unsunk_hits.append((i, j))
                        elif ship.direction == "vertical" and ship.col == j and ship.row <= i < ship.row + ship.size and not ship.is_sunk():
                            unsunk_hits.append((i, j))
        return unsunk_hits
    
    def update_board(self):
        if self.display:
            self.text_area.delete(1.0, tk.END)
            column_letters = "  A B C D E F G H I J"
            self.text_area.insert(tk.END, column_letters + "\n")
            self.text_area.insert(tk.END, "  " + "+-+-+-+-+-+-+-+-+-+-+\n")
            for i, row in enumerate(self.boards[self.current_step]):
                self.text_area.insert(tk.END, f"{i+1:2}| {'|'.join(row)} |\n")
                self.text_area.insert(tk.END, "  " + "+-+-+-+-+-+-+-+-+-+-+\n")
            for ship in self.ships:
                if not ship.is_sunk():
                    self.text_area.insert(tk.END, f"\nShip of size {ship.size} is still alive.\n")
            self.moves_label.config(text=f"Score: {self.score}, Turn: {self.current_step +1}/{len(self.boards)}")
            self.slider.config(to=len(self.boards) - 1)
            self.slider.set(self.current_step)
            if self.game_over:
                self.text_area.insert(tk.END, "\nAll ships sunk! Game over.\n")
                
    def next_step(self):
        if self.current_step >= len(self.boards):
            return
        alive_ships = self.get_alive_ships()
        row, col = self.agent.make_move(self.boards[self.current_step], alive_ships,first_unsunk_hit=self.get_unsunk_hits())
        self.moves.append((row, col))
        self.score += 1
        
        if self.boards[self.current_step][row][col] == 'O':
            self.boards[self.current_step][row][col] = 'X'  # Hit
            self.agent.last_hit = (row, col)  # Update last_hit
            for ship in self.ships:
                if ship.direction == "horizontal" and ship.row == row and ship.col <= col < ship.col + ship.size:
                    ship.hits += 1
                    """
                    if ship.is_sunk():
                        print(f"\nShip of size {ship.size} sunk!\n")
                        print("score : ",self.score)
                    """
                elif ship.direction == "vertical" and ship.col == col and ship.row <= row < ship.row + ship.size:
                    ship.hits += 1
                    """
                    if ship.is_sunk():
                        print(f"\nShip of size {ship.size} sunk!\n")
                        print("score : ",self.score)
                    """
        else:
            #print("miss at",row,col)
            self.boards[self.current_step][row][col] = '-'  # Miss
            self.agent.last_hit = None  # Update last_hit

        # Check if all ships are sunk before adding a new board and incrementing current_step
        if not all(all(cell != 'O' for cell in row) for row in self.boards[self.current_step]):
            self.boards.append(copy.deepcopy(self.boards[self.current_step]))
            self.current_step += 1
        self.update_board()

    def on_scale_move(self, value):
        try:
            new_step = int(round(float(value)))
            if 0 <= new_step < len(self.boards) and new_step != self.current_step:
                self.current_step = new_step
                self.update_board()
        except ValueError:
            pass

    def next_step_button(self):
        if self.current_step < len(self.boards) - 1:
            self.current_step += 1
            self.update_board()

    def prev_step_button(self):
        if self.current_step > 0:
            self.current_step -= 1
            self.update_board()
            
def get_agent_average(agent_class, num_games=100, ships=[5, 4, 3, 3, 2], display=False):
    total_score = 0
    for _ in range(num_games):
        agent = agent_class(10)  # Create a new agent for each game
        app = BoardGUI(None, agent, size=10, ships=ships, display=False)
        total_score += app.score

    average_score = total_score / num_games
    print("Average score:", average_score)
    
    if display:
        # Display a new game with tkinter
        root = tk.Tk()
        root.title("Battleship Game")
        agent = agent_class(10)  # Create a new agent for the displayed game
        app = BoardGUI(root, agent, size=10, ships=ships, display=True)
        root.mainloop()

    return average_score

def main():
    
    print("Random agent :")
    rand_average_score = get_agent_average(RandomAgent,num_games=1, ships=[5,4,3,3,2], display=True)  # Pass the class, not an instance
    
    print("HitNearby agent :")
    hitNearby_average_score = get_agent_average(HitNearbyAgent,num_games=1000, ships=[5,4,3,3,2], display=True)  
    
    print("HitNearby agent :")
    hitAllNearby_average_score = get_agent_average(HitAllNearbyAgent,num_games=1000, ships=[5,4,3,3,2], display=True)  
    
    print("DirectionalHit agent :")
    DirectionalHitagent_average_score = get_agent_average(DirectionalHitAgent,num_games=1000, ships=[5,4,3,3,2], display=True)  
    
    print("DirectionalSunk agent :")
    DirectionalSunkagent_average_score = get_agent_average(DirectionalSunkAgent,num_games=1000, ships=[5,4,3,3,2], display=True)  
    
    
if __name__ == "__main__":
    main()
