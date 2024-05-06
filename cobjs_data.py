import random
import json
from dataclasses import dataclass
from typing import Optional
import re
import pandas as pd
import numpy as np

all_simple_objects = [
    "pencil",
    "notebook",
    "pen",
    "cup",
    "plate",
    "jug",
    "mug",
    "puzzle",
    "textbook",
    "leash",
    "necklace",
    "bracelet",
    "bottle",
    "ball",
    "envelope",
    "lighter",
    "bowl"
]

all_colors = [
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "brown",
    "magenta",
    "fuchsia",
    "mauve",
    "teal",
    "turquoise",
    "burgundy",
    "silver",
    "gold",
    "black",
    "grey",
    "purple",
    "pink"
]

all_simple_colors = [
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "brown",
    "black"
]


surfaces = ["table", "desk", "floor"]
subject_starts = ["there is", "you see", "I see"]

def generate_items_text(chosen_colors, chosen_objects):
  items = []
  for color, obj in zip(chosen_colors, chosen_objects):
    item = "a{} {} {}".format("n" if color[0] in "aeiou" else "", color, obj)
    items.append(item)

  if len(chosen_objects) == 2:
    items_text = " and ".join(items)
  else:
    items_text = ", ".join(items[:-1]) + ", and " + items[-1]
  return items_text

def rand_bool():
  return random.random() > 0.5

def gen_count_scores(true_count, max_number=6):
  scores = {}
  for i in range(max_number+1):
    scores[str(i)] = (1 if i == true_count else 0)
    scores[textify_number(i)] = (1 if i == true_count else 0)
  return scores

def gen_color_scores(true_color):
  return {c: (1 if true_color == c else 0) for c in all_colors}

def pluralize_obj(obj):
  if obj == "sheet of paper":
    return "sheets of paper"
  elif obj == "dog leash":
    return "dog leashes"
  elif obj == "pair of sunglasses":
    return "pairs of sunglasses"
  else:
    return obj + "s"

@dataclass
class Example:
    surface: str
    ss: str
    objects: list
    colors: list
    query_idx: int
    invalid_obj: Optional[str] = None
        
    def copy(self, surface, ss, objects, colors, query_idx=0, invalid_obj=None):
        return Example(str(surface), str(ss), list(objects), list(colors), int(query_idx), invalid_obj)
    
    def self_copy(self):
        return self.copy(self.surface, self.ss, self.objects, self.colors, self.query_idx, self.invalid_obj)
        
    def __str__(self):
        query_object = self.objects[self.query_idx]
        items = generate_items_text(self.colors, self.objects)
        temp = "Q: On the {surf}, {ss} {items}. What color is the {q_object}?\nA:"
        
        if self.invalid_obj is not None:
            query_object = self.invalid_obj
            
        return temp.format(surf=self.surface, ss=self.ss, items=items, q_object=query_object)
    
    def __repr__(self):
        return str(self)
    
    def get_label(self, uppercase=True):
        if self.invalid_obj != None:
            label= random.choice( list(set(all_simple_colors).difference(self.colors)) )
        else:
            query_idx = self.query_idx
            label = self.colors[query_idx]
        label = label.lower()
        if uppercase:
            label = label.title()
        return " "+label
    
    def with_label(self, uppercase=True, override_idx=None):
        if override_idx == None and self.invalid_obj != None:
            label= random.choice( list(set(all_simple_colors).difference(self.colors)) )
        else:
            if override_idx != None:
                query_idx = override_index
            else:
                query_idx = self.query_idx
            label = self.colors[query_idx]
        if uppercase:
            label = " "+label.title()
        else:
            label = " "+label.lower()
        return str(self)+label
    
    def set_query(self, idx):
        assert idx < len(objects)
        self.query_idx = idx
        
    def min_pair_obj(self, manual_idx=None, manual_obj=None):
        #pass manual idx or obj but not both
        if manual_obj in self.objects:
            manual_idx = self.objects.index(manual_obj)
        idx = manual_idx
        assert idx !=self.query_idx
        if idx is None:
            avail_idxs = list(range(len(self.objects)))
            print(self.query_idx, avail_idxs, self.objects)
            avail_idxs.remove(self.query_idx)
            idx = random.choice(avail_idxs)
        
        return self.copy(self.surface, self.ss, self.objects, self.colors, idx)
    
    def n_objs(self, n=3):
        exs = []
        for i in range(n):
            if i == self.query_idx:
                continue
            exs.append(self.min_pair_obj(manual_idx=i))
                
        return exs
            
        
    def min_pair_color(self, manual_idx=None, color_choice=None):
        new_colors = list(self.colors)
        idx = manual_idx
        if idx is None:
            idx = random.choice(range(len(self.colors)))
        
        if color_choice is None:
            color_choice = random.choice( list(set(all_simple_colors).difference(new_colors)) )
            
        new_colors[idx] = color_choice
        return self.copy(self.surface, self.ss, self.objects, new_colors, self.query_idx)
    
    def n_colors(self, n, manual_idx = None):
        new_colors = list(self.colors)
        if manual_idx is None:
            manual_idx = self.query_idx
        exs = []
        color_choices = list(set(all_simple_colors).difference(new_colors))
        random.shuffle(color_choices)
        color_choices = color_choices[:n]
        for color_choice in color_choices:
            exs.append(self.min_pair_color(manual_idx, color_choice))
            
        return exs
    
    def min_pair_invalid(self, obj_choice=None):
        obj = obj_choice
        if obj is None:
            obj = random.choice( list(set(all_simple_objects).difference(self.objects)) )
        
        assert obj not in self.objects and obj is not None
        
        return self.copy(self.surface, self.ss, self.objects, self.colors, self.query_idx, obj)
    
    def n_invalid(self, n):
        exs = []
        objs = list(set(all_simple_objects).difference(self.objects))
        random.shuffle(objs)
        objs = objs[:n]
        for o in objs:
            exs.append(self.min_pair_invalid(o))
        return exs
    
    def get_uplabel(self, color):
        return ' '+color.title()
    
    def obj_color_pair(self, idx):
        return self.objects[idx], self.colors[idx]
    
    def get_viable_preds(self, uppercase=True):
        if uppercase:
            return [' '+label.title() for label in self.colors]
        return [' '+label for label in self.colors]
    

    
class NShotPrompt:
    
    def __init__(self, uppercase=True):
        self.nshots = [] #list of Example
        self.uppercase=uppercase
        
    def __str__(self):
        prompt = ''
        if len(self.nshots)==1:
            return str(self.nshots[0])
        
        for ex in self.nshots[:-1]:
            prompt+=ex.with_label(uppercase=self.uppercase)+'\n'
        
        prompt+=str(self.nshots[-1])
        return prompt
    
    def add_shot(self, ex: Example):
        self.nshots.append(ex)
        
    def get_label(self, uppercase=True):
        return self.nshots[-1].get_label(uppercase)
        
    def viable_answers(self, uppercase=True):
        viable = []
        for ex in self.shots:
            viable+=ex.get_viable_preds(uppercase)
        return viable
    
    def make_add_ex(self, n_objs=3, exclude_seen_objs=False, exclude_seen_colors=False, surface=None, ss=None):
        excluded_cols = []
        excluded_objs = []
        for ex in self.nshots:
            if exclude_seen_objs:
                excluded_objs+=ex.objects
            if exclude_seen_colors:
                excluded_cols+=ex.colors
                
        obj_list = set(all_simple_objects)
        col_list = set(all_simple_colors)
        obj_list = list(obj_list.difference(set(excluded_objs)))
        col_list = list(col_list.difference(set(excluded_cols)))
        if exclude_seen_objs and len(obj_list) < n_objs:
            raise Exception("There are not enough unseen objects to make a new example")
        
        if exclude_seen_colors and len(col_list) < n_objs:
            raise Exception("There are not enough unseen colors to make a new example")
        
        chosen_colors = random.sample(col_list, n_objs)
        chosen_objects = random.sample(obj_list, n_objs)
        if ss is None:
            ss = random.choice(subject_starts)
        if surface is None:
            surface = random.choice(surfaces)
        
        query_idx = random.choice(list(range(n_objs)))
        ex = Example(str(surface), str(ss), list(chosen_objects), list(chosen_colors), query_idx)
        self.add_shot(ex)
        
        
    @classmethod
    def parse(cls, prompt: str, uppercase=True):
        prompts = prompt.split('Q:')[1:]
        nshots = []
        #print(prompts)
        if len(prompts)>1:
            ans = prompt.split("A: ")[1]
            #print("ANS", ans)
            if ans[0].isupper():
                uppercase=True
            else:
                uppercase=False
            
        
        def parse_single_ex(ex):
            """
            Q: On the table, there is a purple textbook, a brown bowl, and a green cup. What color is the ball?
            A:
            """
            words = ex.split()
            objs = []
            cols = []
            
            #the surface: table, desk, floor
            surface = re.search( r'On the (table|desk|floor)', ex).group(0).split()[2]
            #subject_starts
            starts = r'({s1}|{s2}|{s3})'.format(s1=subject_starts[0], s2=subject_starts[1], s3=subject_starts[2])
            ss = re.search( starts, ex).group(0)
            objsearch = '[ ,.?]|'.join(all_simple_objects)
            objsearch='('+objsearch+'[ ,.?])'
            #print(objsearch)
            for obj_match in re.findall(objsearch, ex):
                objs.append(obj_match[:-1])
                
            colsearch = '|'.join(all_simple_colors)
            colsearch= '('+colsearch+')'
            
            for color in re.findall(colsearch, ex):
                cols.append(color)
                    
            query_obj = objs[-1]
            query_idx=0
            invalid_obj = None
            try: 
                query_idx = objs.index(query_obj) #get the first occurrence
                #print('try', query_idx, objs[query_idx])
            except ValueError as e:
                invalid_obj = query_obj
            #print('objs', objs)
            #print("cols", cols)
            objs = objs[:-1]
            return Example(str(surface), str(ss), list(objs), list(cols), int(query_idx), invalid_obj)
            
        
        for prompt in prompts:
            nshots.append(parse_single_ex("Q: "+prompt))
            
        nshot_prompt = NShotPrompt(uppercase=uppercase)
        nshot_prompt.nshots=nshots
        return nshot_prompt


