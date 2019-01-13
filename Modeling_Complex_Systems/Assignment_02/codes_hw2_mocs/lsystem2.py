# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 09:18:40 2018

@author: mgreen13
"""
import numpy as np
import copy 
import turtle 



def Lsystem(axiom,rules,runs,rot,dist):
    """ 
    DESCRIPTION: Lsystem calls on lsysExpand and lsysDrawl, passing in required inputs. 
    
    INPUTS: axiom = initial state of system
            rules = rules that expansion will follow
            runs = number of iterations
            rot = rotation in degrees of turtle
            dist = distance the turtle will step
   
    OUTPUTS: all_str: A list of instances of the system from each iteration
    """
    carl = turtle.Turtle()
    turtle.mode("logo")
    turtle.speed(0)
    all_str = []
    # BEFORE = AXIOM FOR FIRST COMPARISON
    before = copy.deepcopy(axiom)
    for run in range(runs):
        # AFTER = EXPANDED(BEFORE)
        after = lsysExpand(axiom,before,rules)
        all_str.append(copy.deepcopy(after))
        # AFTER -- > BEFORE FOR NEXT ITERATION
        before = after
    string = (all_str[len(all_str)-1])
    lsysDraw(string,carl,rot,dist)
    return(all_str)
    
def lsysDraw(lsys,carl,rot,dist):
    """ 
    DESCRIPTION: lsysDraw uses the turtle program module to draw the L-System 
    
    INPUTS: lsys = string to be drawn
            carl = our dear turtle, carl
            rot = rotation in degrees of turtle
            dist = distance the turtle will step
   
    OUTPUTS: NONE
    """
    history = []
    for rule in lsys:
        print(rule)
        if rule == "F":
            carl.forward(dist)
        elif rule == "D":
            carl.back(dist)
        elif rule == "C":
            carl.circle(2)
        elif rule == "|":
            carl.forward(dist)
        elif rule == "[":
            angle = carl.heading()
            position = [carl.xcor(),carl.ycor()]
            history.append((angle,position))
        elif rule == ']':
            angle,position = history.pop()
            carl.penup()
            carl.setposition(position)
            carl.setheading(angle)
            carl.pendown()
                                        # IMPLEMENTING INTEGER + / - FOR CHANGE IN DIRECTION
        if len(rule) == 2:
            time = rule[0]
            if rule[1] == "-":
                for i in range(int(time)):
                    carl.right(rot)
            if rule[1] == "+":
                for i in range(int(time)):
                    carl.left(rot)
                
        if rule == "+":
            carl.left(rot)
        if rule == "-":
            carl.right(rot)
          
    turtle.done()
    return

def lsysExpand(axiom,before,rules):
    """
    DESCRIPTION: lsysExpand expands the axiom and future iterates using the system rules 
    
    INPUTS: axiom = initial state of system
            before = the list of elements before the rule is applied
            rules = the rule to be applied
   
    OUTPUTS: after = the list of elements after the rule has been applied.
    """
    for ind,ele in enumerate(before):               # LOOP OVER CHARACTERS IN BEFORE
        if ele == axiom[0]:                         # IF ELEMENT == AXIOM
            before[ind] = copy.deepcopy(rules)    # REPLACE ELEMENT WITH RULE IN LIST
    after = []
    for index,element in enumerate(before):        # GET RID OF LIST WRAPPERS AROUND RULE INSTANCES IN BEFORE
        after.extend(element)
    return(after)


# IMPLEMENTATION EXMAPLE
axiom = ["F"]
# FLAKE TWIG RULES
rules = ["F","[","+","F","]","F","[","-","F","]","F"]
# EXAMPLE USE OF INTEGER MULTIPLICATION
rules2 = ["F","[","F","2-","F","C","]","5+","[","F","+","F","C","+","F","C","]"]
# BIRDS NEST
rules3 =  ["F","[","F","-","F","C","]","+","[","F","+","F","C","+","F","C","]"]
# Calling L-System will produce the image resulting from the given axiom and rule
final = Lsystem(axiom,rules3,3,18,30)