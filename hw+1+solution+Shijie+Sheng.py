
# coding: utf-8

# In[3]:

def is_mcnugget_number(num, a = 6, b = 9, c = 20):
    for i in range(num//a + 1):
        for j in range(num//b + 1):
            for k in range(num//c + 1):
                if (a*i +b*j +c*k == num):
                    return True;
                
    return False
            
def main():
    candidate = 1
    biggest = 0
    counter = 0
    six = 6
    
    while(counter < six):
        if(is_mcnugget_number(candidate)):
            counter += 1
        else:
            biggest = candidate
            counter = 0
        candidate += 1
    print("The largest number of nuggets that you cannot buy is {}".format(biggest))
if __name__ == "__main__":
    main() 


# In[4]:

menu = {'1' : 'TOOLO', '2' : 'TOOHI', '3' : 'CORRECT'}
        
def bisector(lower, upper):
    return (lower + upper) // 2
    
def greeting():
    print("\tWelcome to Guess My Number!\n\n")
    print("In this version of the game you will guess a number and I'll try to guess it\n")
    
def get_response():
    choice = input("How did I do: 1) too low, 2) too high, or 3) correct?")
    
    return choice
    
def main():
    greeting()
    
    lower = 1
    upper = 100
    tries = 0
    game_over = False
    
    while(not game_over):
        guess = bisector(lower, upper)
        print("My guess is: {}".format(guess))
        user_response = get_response()
        if menu[user_response] == 'TOOHI':
            upper = guess
        elif menu[user_response] == 'TOOLO':
            lower = guess
        else:
            game_over = True
            break
        tries += 1
    
    print("I correctly guessed your number {} in {} tries".format(guess, tries))
    
if __name__ == "__main__":
    main()


# In[ ]:



