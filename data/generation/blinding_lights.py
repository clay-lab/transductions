# blinding_lights.py
# 
# Produce sequences of "on" and "off" lights, paired with a quantified expression
# describing how many lights are on.
# 
#   💡 💡 💡 ⚫️   ->    most lights are on
#   💡 💡 💡 💡   ->    all lights are on
#   💡 💡 ⚫️ ⚫️   ->    some lights are on
#   ⚫️ ⚫️ ⚫️ ⚫️   ->    no lights are on
#   💡 ⚫️ ⚫️ ⚫️   ->    few lights are on
# 
# From this we can test generalization to longer (or shorter) lengths, out-of-order 
# sequences (i.e., train on having all the "on" lights be first and then test what 
# happens when the lights are out of order)

import numpy as np
from emoji import UNICODE_EMOJI

def get_quantifier(on: int, total: int):

  if on == 0:
    return "None"
  elif on == total:
    return "All"
  else:
    prop = float(on) / float(total)
    if prop > 0.5:
      return "Most"
    else:
      return "Some"

def add_space(text):
    return ''.join(' ' + char if char in UNICODE_EMOJI else char for char in text).strip()

def generate_sequences(outfile: str):

  ON = u"💡"
  OFF = u"⚫️"

  MAX_LENGTH = 10
  NUM_SENTENCES = 10_000

  lengths = np.random.randint(1, MAX_LENGTH, size=NUM_SENTENCES)
  num_on = np.random.randint(0, lengths + 1)

  with open(outfile, 'w') as f:
    f.write("source\ttransformation\ttarget\n")
  
  with open(outfile, 'a') as f:
    for i in range(NUM_SENTENCES):
      on_lights = ON * num_on[i]
      off_lights = OFF * (lengths[i] - num_on[i])
      lights = add_space(on_lights + off_lights)
      # print([ord(l) for l in lights.split()])
      quant = get_quantifier(num_on[i], lengths[i])
      target = f"{quant} of the lights are on"
      f.write(f"{lights}\tquant\t{target}\n")

def main():
  generate_sequences("data/raw/lights.tsv")

if __name__ == "__main__":
  main()
