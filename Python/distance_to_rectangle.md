---
title: "Python Exercise: Distance to Rectangle"
date: 2017-06-28
categories: Python
tags: python
slug: rectangle
author: Dr. Juan Camilo Orduz
summary: In this first post we get started with a small python script to explore the basic capabilities of Pelican.
---

In this first post I wanted to explore the basics of blogging [blogdown](https://bookdown.org/yihui/blogdown/). I treat an example of a little python challenge which I encountered in first my job hunt process. I particularly like it because it is a geometric problem.   

# Problem

*Write a function that tests if a point falls within a specified distance "dist" of any part of a solid, 2D rectangle.  The rectangle is specified by the bottom left corner, a width, and a height.* 

## Suggested Solution

We define a class that represents a Point. Each Point has an x and y coordinate.

```python  
class Point(object):
    
    def __init__(self, init_x=0.0, init_y=0.0):
        self.x = init_x
        self.y = init_y
```

We define a class that represents a Rectangle. Its position is determined by the Point at the bottom left corner. The dimensions of the Rectangle is determined by its width and height.

```python
 class Rectangle(object):
    
    def __init__(self, bottom_left_x=0.0, bottom_left_y=0.0, init_width=0.0, init_height=0.0):
        self.bottom_left = Point(bottom_left_x, bottom_left_y)
        self.width = init_width
        self.height = init_height
```

This function idicates whether a given point lies within a distance dist of a given Rectangle, returning a boolen value. That is, if $A\subset \mathbb{R}^2$ denotes the given rectangle, we want to compute the indicatior function of the open set

$$
U_{dist}:= \\{ p \in\mathbb{R}^2 : \exists q \in R \quad \text{such that} \quad ||p-q|| < dist \\}.
$$

First function verifies if the x-coordinate of the Point is at a distance less than dist. If it is then it verifies the y-coordinate. For the y-coordinate it evaluates two cases:

   1. It checks if its outside the Rectangle but still within a distance less that dist (checks to the left and to the right).

   2. It checks if it is inside the Rectangle. 

```python
import math

def is_point_within_dist_of_rect(rect=Rectangle(), point=Point(), dist=0.0):
    
    if((rect.bottom_left.x - dist)< point.x and point.x < (rect.bottom_left.x + rect.width + dist)):
        
        if(point.x < rect.bottom_left.x):
            
            a = rect.bottom_left.x - point.x
            y_max = rect.bottom_left.y + rect.height + math.sqrt(dist**2-a**2)
            y_min = rect.bottom_left.y - math.sqrt(dist**2-a**2)
            
            if((y_min < point.y) and point.y < y_max):
                return True
            else:
                return False
            
        elif(point.x < (rect.bottom_left.x + rect.width)):
            
            y_max = rect.bottom_left.y + rect.height + dist
            y_min = rect.bottom_left.y - dist
            
            if((y_min < point.y) and point.y < y_max):
                return True
            else:
                return False
            
        else:

            a = rect.bottom_left.x+rect.width - point.x
            y_max = rect.bottom_left.y + rect.height + math.sqrt(dist**2-a**2)
            y_min = rect.bottom_left.y - math.sqrt(dist**2-a**2)
            
            if((y_min < point.y) and point.y < y_max):
                return True
            else:
                return False
    
    return False
    
```

## Eamples

We consider square of side 2 with the center of mass at the origin.

```python
rectangle = Rectangle(-1,-1,2,2)
```

We check that the origin is in the square.

```python
point_1 = Point(0,0)

is_point_within_dist_of_rect(rectangle, point_1, dist=1)
```


    True


We check that the upper right corner is in the square.

```python
point_2 = Point(1,1)

is_point_within_dist_of_rect(rectangle, point_2, dist=1)
```


    True


We check a point outside a the square.

```python
point_3 = Point(0,3)

is_point_within_dist_of_rect(rectangle, point_3, dist=1)
```

    False

Now we consider a limit case. First we define:

```python
threshold_value= 1 + math.sqrt(2)/2
epsilon = 0.0001
```

We consider two cases:

- We check a point close (outside) to the region boundary.

```python

point_4 = Point(threshold_value + epsilon, threshold_value + epsilon)

is_point_within_dist_of_rect(rectangle, point_4, dist=1)
```


    False


- We check a point close (inside) to the region boundary.

```python
point_5 = Point(threshold_value - epsilon, threshold_value - epsilon)

is_point_within_dist_of_rect(rectangle, point_5, dist=1)
```


    True
