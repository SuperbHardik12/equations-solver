import mysql.connector as c
import math as m
import matplotlib
from random import random as r
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtCore import QRect
from PyQt5.QtWidgets import QMessageBox, QLineEdit
from skimage import measure
from PyQt5 import QtWidgets, QtCore, QtGui
import re
import sys


def show_error_dialog(message):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setText("Error")
    msg.setInformativeText(message)
    msg.setWindowTitle("Error Dialog")
    msg.exec_()


def update_function1(newhistory, userid, answer, is_continued):
    if is_continued:
        cur.execute('use equation_solver')
        cur.execute(f'select history from history where useride={userid};')
        data = cur.fetchone()
        newhistory = data[0] + '$' + newhistory
        cur.execute(f'select answer from history where useride={userid};')
        data1 = cur.fetchone()
        newanswer = data1[0] + '$' + answer
        cur.execute('update history set history=%s, answer=%s where useride=%s;', (newhistory, newanswer, userid))
        connect.commit()
    else:
        cur.execute('use equation_solver')
        cur.execute(f'select history from history where useride={userid};')
        data = cur.fetchone()
        newhistory = data[0] + '#' + newhistory
        cur.execute(f'select answer from history where useride={userid};')
        data1 = cur.fetchone()
        newanswer = data1[0] + '#' + answer
        cur.execute('update history set history=%s, answer=%s where useride=%s;', (newhistory, newanswer, userid))
        connect.commit()


def insert_function(name, pwd):
    global userid
    cur.execute('use equation_solver;')
    cur.execute('select useride from userid;')
    data = cur.fetchall()
    max = 0
    for i in data:
        for j in i:
            if int(j) > max:
                max = int(j)
    uid = max + 1
    userid = uid
    sno = uid
    cur.execute(f'insert into userid values (%s, %s, %s, %s);', (sno, name, pwd, uid))
    cur.execute(f'insert into history values ({uid}, " ", " ");')
    connect.commit()


def plot_3d_graph(equation, x_range=(-10, 10), y_range=(-10, 10), num_points=100):
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)

    try:
        Z = eval(equation.replace('^', '**').replace('x', 'X').replace('y', 'Y'))

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_title(f'Plot of {equation}')

        ax.set_xlim(None)
        ax.set_ylim(None)
        ax.set_zlim(None)

        plt.show()
    except Exception as e:
        print(f"Error evaluating the equation: {e}")


def plot_implicit_3d(equations, x_range=(-10, 10), y_range=(-10, 10), z_range=(-10, 10), num_points=100):
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    z = np.linspace(z_range[0], z_range[1], num_points)
    X, Y, Z = np.meshgrid(x, y, z)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.viridis(np.linspace(0, 1, len(equations)))  # Use colormap to get different colors

    for idx, equation in enumerate(equations):
        eqn = equation
        eqn = eqn.split('=')[0] + '-(' + eqn.split('=')[1] + ')'
        eqn = eqn + '-1000'
        processed_eq = preprocess_equation(eqn)

        try:
            values = eval(processed_eq.replace("^", "**").replace('x', 'X').replace('y', 'Y').replace('z', 'Z'))

            # Use the marching cubes algorithm to extract the surface
            verts, faces, _, _ = measure.marching_cubes(values, level=0, spacing=(x[1] - x[0], y[1] - y[0], z[1] - z[0]))

            # Plot the surface with a specific color
            ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], color=colors[idx], lw=1, alpha=0.6, label=f'{equation}')

        except Exception as e:
            print(f"Error evaluating the equation '{equation}': {e}")

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('Plot of 3D Equations')

    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    ax.set_zlim(z_range[0], z_range[1])

    plt.legend(loc='best')
    plt.show()
def preprocess_equation(equation):
    # Handle coefficient and exponent syntax
    equation = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', equation)
    equation = re.sub(r'([a-zA-Z])\^(\d+)', r'\1**\2', equation)
    return equation


def plot_implicit(equations, x_range=(-10, 10), y_range=(-10, 10), num_points=2000, zoom_x=(-50, 50),
                  zoom_y=(-50, 50)):
    plt.figure(figsize=(8, 6))

    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)
    labels = []
    for eq in equations:
        eqn = eq.split('=')[0] + '-(' + eq.split('=')[1]+')'
        processed_eq = preprocess_equation(eqn)

        try:
            Z = eval(processed_eq.replace('x', 'X').replace('y', 'Y'))
            contour = plt.contour(X, Y, Z, levels=[0])
            labels.append(eq)

        except Exception as e:
            print(f"Error evaluating the equation '{eq}': {e}")

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Plot of Equations')
    plt.gca().set_aspect('equal')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlim(zoom_x[0], zoom_x[1])
    plt.ylim(zoom_y[0], zoom_y[1])
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    if labels:
        plt.legend(labels, loc='best')
    plt.show()

matplotlib.use('Qt5Agg')

userid = [""]
connect = c.connect(host='localhost', user='root', passwd='1234')
cur = connect.cursor()


def plot_multiple_3d_planes(equations, variables):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create a grid of points
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)

    # Plot each plane
    for equation in equations:
        coefficients = equation['coefficients']
        constant = equation['constant']
        # Compute Z using the equation
        Z = (-coefficients[0] * X - coefficients[1] * Y + constant) / coefficients[2]

        # Plot the plane
        ax.plot_surface(X, Y, Z, alpha=0.6, rstride=100, cstride=100)

    ax.set_xlabel(variables[0])
    ax.set_ylabel(variables[1])
    ax.set_zlabel(variables[2])
    ax.set_title('3D Plane Representation')
    plt.get_current_fig_manager().window.showMaximized()
    plt.show()


def randomized(n):
    a = list()
    for i in range(1, n + 1):
        a.append(int(r() * 1000))
    return a


def find_right(a):
    a = brackets_check(a)
    checker = ['+', '-', '=']
    ind = a.index('=')
    a = a[:ind:] + '+' + a[ind] + '+' + a[ind + 1::]
    summ = 0
    for i in range(len(a)):
        n = ""
        if a[i] in checker and a[i - 1].isdigit():
            for j in range(i - 1, -1, -1):
                if not a[j].isdigit() and a[j] != '.':
                    n += a[j]
                    break
                n += a[j]
            if i <= ind:
                summ -= int(n[::-1])
            else:
                summ += int(n[::-1])
    return summ


def find_coefficientsl(a, variable):
    a = brackets_check(a)
    left = a.split('=')[0] + '='
    right = a.split('=')[1] + '='
    summ1, summ2, summ3 = 0, 0, 0
    for j in range(len(left)):
        t1, t2, t3 = "", "", ""
        if left[j] == variable:
            temp = j - 1
            for k in range(j - 1, -1, -1):
                if not left[k].isdigit() and left[k] != '.':
                    temp = k
                    break
                t2 += left[k]
            if left[temp] == '-':
                summ2 -= int(t2[::-1]) if t2 else 1
            else:
                summ2 += int(t2[::-1]) if t2 else 1
    for j in range(len(right)):
        t1, t2, t3 = "", "", ""
        if right[j] == variable:
            temp = j - 1
            for k in range(j - 1, -1, -1):
                if not right[k].isdigit() and right[k] != '.':
                    temp = k
                    break
                t2 += right[k]
            if right[temp] == '-':
                summ2 += int(t2[::-1]) if t2 else 1
            else:
                summ2 -= int(t2[::-1]) if t2 else 1
    return summ2


def solve_linear1(a, b):
    try:
        x = np.linalg.solve(a, b)
        return x
    except np.linalg.LinAlgError:

        return "The system of equations entered is inconsistent and hence does not have any solutions."


def solve_linear(equations_list, user):
    result = ""
    v = []

    # Extract unique variable names from equations
    for equation in equations_list:
        for i in equation:
            if i.isalpha() and i not in v:
                v.append(i)

    mat_ovr = []
    mat_ovr2 = []

    for equation in equations_list:
        mat1 = []
        for i in v:
            mat1.append(find_coefficientsl(equation, i))
        mat_ovr.append(mat1)
        mat_ovr2.append(find_right(equation))

    # Solve the linear system of equations
    j = solve_linear1(np.array(mat_ovr), np.array(mat_ovr2))

    # Format the results
    for i in range(len(v)):
        result += f'{v[i]}: {j[i]}\n'

    update_function1(' '.join(equations_list), user, result.strip(), False)

    # Prepare equations for plotting
    equations_to_plot = []
    equations_to_plot1 = []
    for equation in equations_list:
        coefficients = [find_coefficientsl(equation, i) for i in v]
        constant = find_right(equation)
        equations_to_plot.append({'coefficients': coefficients, 'constant': constant})
        equations_to_plot1.append(equation)

    # Plotting based on the number of variables
    if len(v) <= 2:
        plot_implicit(equations_to_plot1,
                        x_range=(-1000, 1000), y_range=(-1000, 1000),
                        zoom_x=(-100, 100), zoom_y=(-100, 100))
    elif len(v) == 3:
        plot_multiple_3d_planes(equations_to_plot, v)
    else:
        show_error_dialog("Equations containing more than 3 variables cannot be plotted!")

    return result.strip()



def solve_quadratic1(equations, user):
    final_result = ""
    plt.figure()
    for equation in equations:
        variables = []
        for i in range(len(equation)):
            if equation[i] == '^':
                if equation[i - 1].isalpha() and equation[i - 1] not in variables:
                    variables.append(equation[i - 1])
        if not variables:
            final_result += "No valid variables found in equation: " + equation + "\n"
            continue

        ans = "For equation " + equation + "\n"
        coefficients = []
        solutions = []
        for i in range(len(variables)):
            coef = find_coefficients(equation, variables[i])
            if coef:
                coefficients.append(coef)
                solution = solve_quadratic(coef, variables[i])[0]
                solutions.append([solve_quadratic(coef, variables[i])[1], solve_quadratic(coef, variables[i])[2]])
                update_function1(equation, userid, solution, i)
                ans += solution
            else:
                ans += f"Unable to find coefficients for variable {variables[i]}\n"
    if len(variables) <= 2:  # 2D case
        plot_implicit(equations,
                        x_range=(-1000, 1000), y_range=(-1000, 1000),
                        zoom_x=(-100, 100), zoom_y=(-100, 100))
    elif len(variables) == 3:  # 3D case
        plot_implicit_3d(equations, x_range=(-50, 50),
                            y_range=(-50, 50), z_range=(-50, 50), num_points=300)
    else:
            show_error_dialog("Equations containing more than 3 variables can not be plotted!")
    final_result += ans

    return final_result


def solve_quadratic(d, v):
    ans = "Roots for " + v + "\n"
    a, b, c = d[0], d[1], d[2]
    dis = b ** 2 - 4 * a * c
    if dis >= 0:
        r_1 = (-b - m.sqrt(dis)) / (2 * a)
        r_2 = (-b + m.sqrt(dis)) / (2 * a)
        ans = ans + "Root 1: " + str(r_1) + "\nRoot 2: " + str(r_2) + '\n'
    else:
        r_1 = (-b - m.sqrt(-dis) * 1j) / (2 * a)
        r_2 = (-b + m.sqrt(-dis) * 1j) / (2 * a)
        ans = ans + "Root 1: " + str(r_1) + "\nRoot 2: " + str(r_2) + '\n'
    return [ans, r_1, r_2]


def brackets_check(a):
    a = '+' + a + '+'
    b = ""
    i = 0
    while i < len(a):
        if a[i] == '-' and a[i + 1] == '(':
            b = b + '-'
            j = i
            while a[j] != ')':
                j = j + 1
            temp = j
            for k in range(i + 1, temp + 1):
                if a[k] == '+':
                    b += '-'
                elif a[k] == '-':
                    b += '+'
                else:
                    b += a[k]
            i = temp
        else:
            b += a[i]
        i = i + 1
    a = b
    a = a.replace('(', '')
    a = a.replace(')', '')
    return a


def find_coefficientsc(a, variable):
    a = brackets_check(a)
    left = a.split('=')[0] + '='
    right = a.split('=')[1] + '='
    summ, summ1, summ2, summ3 = 0, 0, 0, 0
    for j in range(len(left)):
        t1, t2, t3, t4 = "", "", "", ""
        if j != len(left) - 1 and left[j + 1] == '^' and left[j + 2] == '3' and left[j] == variable:
            temp = j - 1
            for k in range(j - 1, -1, -1):
                if not left[k].isdigit():
                    temp = k
                    break
                t4 += left[k]
            if left[temp] == '-':
                summ -= int(t4[::-1]) if t4 else 1
            else:
                summ += int(t4[::-1]) if t4 else 1
        elif j != len(left) - 1 and left[j + 1] == '^' and left[j] == variable:
            temp = j - 1
            for k in range(j - 1, -1, -1):
                if not left[k].isdigit():
                    temp = k
                    break
                t1 += left[k]
            if left[temp] == '-':
                summ1 -= int(t1[::-1]) if t1 else 1
            else:
                summ1 += int(t1[::-1]) if t1 else 1
        elif left[j] == variable:
            temp = j - 1
            for k in range(j - 1, -1, -1):
                if not left[k].isdigit():
                    temp = k
                    break
                t2 += left[k]
            if left[temp] == '-':
                summ2 -= int(t2[::-1]) if t2 else 1
            else:
                summ2 += int(t2[::-1]) if t2 else 1
        elif left[j] in ['+', '-', '=']:
            temp = 0
            if j >= 2 and left[j - 2] == '^' and left[j - 1] == '2' or left[j - 1] == '3':
                continue
            elif left[j - 1].isdigit():
                for k in range(j - 1, -1, -1):
                    if not left[k].isdigit():
                        temp = k
                        break
                    t3 += left[k]
                if left[temp] == '-':
                    summ3 -= int(t3[::-1])
                else:
                    summ3 += int(t3[::-1])
    for j in range(len(right)):
        t1, t2, t3, t4 = "", "", "", ""
        if j != len(right) - 1 and right[j + 1] == '^' and right[j + 2] == '3' and right[j] == variable:
            temp = j - 2
            for k in range(j - 1, -1, -1):
                if not right[k].isdigit():
                    temp = k
                    break
                t4 += right[k]
            if right[temp] == '-':
                summ += int(t4[::-1]) if t4 else 1
            else:
                summ -= int(t4[::-1]) if t4 else 1
        elif j != len(right) - 1 and right[j + 1] == '^' and right[j] == variable:
            temp = j - 2
            for k in range(j - 1, -1, -1):
                if not right[k].isdigit():
                    temp = k
                    break
                t1 += right[k]
            if right[temp] == '-':
                summ1 += int(t1[::-1]) if t1 else 1
            else:
                summ1 -= int(t1[::-1]) if t1 else 1
        elif right[j] == variable:
            temp = j - 1
            for k in range(j - 1, -1, -1):
                if not right[k].isdigit():
                    temp = k
                    break
                t2 += right[k]
            if right[temp] == '-':
                summ2 += int(t2[::-1]) if t2 else 1
            else:
                summ2 -= int(t2[::-1]) if t2 else 1
        elif right[j] in ['+', '-', '=']:
            temp = 0
            if j >= 2 and right[j - 2] == '^' and right[j - 1] == '2' or right[j - 1] == '3':
                continue
            elif right[j - 1].isdigit():
                for k in range(j - 1, -1, -1):
                    if not right[k].isdigit():
                        temp = k
                        break
                    t3 += right[k]
                if right[temp] == '-':
                    summ3 += int(t3[::-1])
                else:
                    summ3 -= int(t3[::-1])
    return [summ, summ1, summ2, summ3]


def solve_cubic(equations, user):
    result = ""

    for equation in equations:
        v = list()
        for i in equation:
            if i.isalpha() and i not in v:
                v.append(i)

        for i in range(len(v)):
            roots = np.roots(find_coefficientsc(equation, v[i]))
            s = f"Roots for {v[i]} in equation '{equation}': \n"

            for j in roots:
                s += str(j) + ", \n"

            update_function1(equation, userid, s, i)
            result += s.strip(", ") + "\n"

        if len(v) <= 2:  # 2D case
            plot_implicit(equations,
                          x_range=(-1000, 1000), y_range=(-1000, 1000),
                          zoom_x=(-100, 100), zoom_y=(-100, 100))
        elif len(v) == 3:  # 3D case
            plot_implicit_3d(equations, x_range=(-50, 50),
                             y_range=(-50, 50), z_range=(-50, 50), num_points=300)
        else:
            show_error_dialog("Equations containing more than 3 variables can not be plotted!")

    return result.strip()


def find_coefficients(a, variable):
    a = brackets_check(a)
    left = a.split('=')[0] + '='
    right = a.split('=')[1] + '='
    summ1, summ2, summ3 = 0, 0, 0
    for j in range(len(left)):
        t1, t2, t3 = "", "", ""
        if j != len(left) - 1 and left[j + 1] == '^' and left[j] == variable:
            temp = j - 1
            for k in range(j - 1, -1, -1):
                if not left[k].isdigit() and left[k] != '.':
                    temp = k
                    break
                t1 += left[k]
            if left[temp] == '-':
                summ1 -= int(t1[::-1]) if t1 else 1
            else:
                summ1 += int(t1[::-1]) if t1 else 1
        elif left[j] == variable:
            temp = j - 1
            for k in range(j - 1, -1, -1):
                if not left[k].isdigit() and left[k] != '.':
                    temp = k
                    break
                t2 += left[k]
            if left[temp] == '-':
                summ2 -= int(t2[::-1]) if t2 else 1
            else:
                summ2 += int(t2[::-1]) if t2 else 1
        elif left[j] in ['+', '-', '=']:
            temp = 0
            if j >= 2 and left[j - 2] == '^' and left[j - 1] == '2':
                continue
            elif left[j - 1].isdigit():
                for k in range(j - 1, -1, -1):
                    if not left[k].isdigit() and left[k] != '.':
                        temp = k
                        break
                    t3 += left[k]
                if left[temp] == '-':
                    summ3 -= int(t3[::-1])
                else:
                    summ3 += int(t3[::-1])
    for j in range(len(right)):
        t1, t2, t3 = "", "", ""
        if j != len(right) - 1 and right[j + 1] == '^' and right[j] == variable:
            temp = j - 2
            for k in range(j - 1, -1, -1):
                if not right[k].isdigit() and right[k] != '.':
                    temp = k
                    break
                t1 += right[k]
            if right[temp] == '-':
                summ1 += int(t1[::-1]) if t1 else 1
            else:
                summ1 -= int(t1[::-1]) if t1 else 1
        elif right[j] == variable:
            temp = j - 1
            for k in range(j - 1, -1, -1):
                if not right[k].isdigit() and right[k] != '.':
                    temp = k
                    break
                t2 += right[k]
            if right[temp] == '-':
                summ2 += int(t2[::-1]) if t2 else 1
            else:
                summ2 -= int(t2[::-1]) if t2 else 1
        elif right[j] in ['+', '-', '=']:
            temp = 0
            if j >= 2 and right[j - 1] == '^' and right[j - 2] == '2':
                continue
            elif right[j - 1].isdigit():
                for k in range(j - 1, -1, -1):
                    if not right[k].isdigit() and right[k] != '.':
                        temp = k
                        break
                    t3 += right[k]
                if right[temp] == '-':
                    summ3 += int(t3[::-1])
                else:
                    summ3 -= int(t3[::-1])
    return [summ1, summ2, summ3]


def create():
    cur.execute('create database if not exists equation_solver;')
    cur.execute('use equation_solver;')
    cur.execute('''create table if not exists userid(
    srno int,
    username varchar(20) not null,
    password varchar(20) not null,
    useride int primary key);''')
    cur.execute('''create table if not exists history(
    useride int,
    foreign key (useride) references userid(useride)
     ON DELETE CASCADE,
    history longtext,
    answer longtext);''')


def update_function(oldpassword, newpassword):
    cur.execute('use equation_solver')
    cur.execute('update userid set password={password} where password={oldpassword} and useride = {userid};'.format(
        password=newpassword, oldpassword=oldpassword, userid=userid))
    connect.commit()


def subtract_matrix(matrix1, matrix2):
    try:
        return matrix1 - matrix2
    except:
        print("invalid!")


def add_matrix(matrix1, matrix2):
    return matrix1 + matrix2


def multiply_matrix(matrix1, matrix2):
    return np.array(matrix1) @ np.array(matrix2)


def matrix_input():
    i = 0
    a = list()
    a.append(list())
    while True:
        x = input("Create a new column or press 'q' to quit: ")
        if x == 'q':
            break
        a[0].append(float(x))
        i += 1
    var = 1
    while var < i:
        a.append(list())
        for k in range(i):
            a[var].append(float(input('Enter a new element: ')))
        x = input('Enter "q" to quit or anything else to continue and add another row: ')
        if x == 'q':
            break
        var += 1
    return a


def solve_equation(matrices, equation):
    result = None
    i = 0
    while i < len(equation):
        if equation[i].isalpha():
            matrix = matrices[equation[i]]
            if result is None:
                result = matrix
            else:
                operation = equation[i - 1]
                if operation == '+':
                    result = add_matrix(result, matrix)
                elif operation == '-':
                    result = subtract_matrix(result, matrix)
                elif operation == '*':
                    result = multiply_matrix(result, matrix)
        i += 1
    return result


def eqn_input():
    a = input('Enter the equation: ')
    v = list()
    for i in a:
        if i.isalpha() and i not in v:
            v.append(i)
    matrices = {}
    for i in v:
        print(f"Input matrix for {i}:")
        matrices[i] = matrix_input()
    return [matrices, a]


def matrix_algebra():
    matrices, equation = eqn_input()
    a = solve_equation(matrices, equation)
    for i in range(len(a) // 2):
        print(a[i])


def login(uname, password):
    global userid
    cur.execute('USE equation_solver')
    cur.execute('SELECT username, password, useride FROM userid;')
    data = cur.fetchall()

    user_dict = {}

    for row in data:
        username, passw, user_id = row
        user_dict[username] = (passw, user_id)

    if uname not in user_dict:
        return False

    stored_password, user_id = user_dict[uname]

    if stored_password != password:
        return False

    userid = user_id
    return True


class Ui_Dialog1(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1200, 800)
        self.bgwidget = QtWidgets.QWidget(Dialog)
        self.bgwidget.setGeometry(QtCore.QRect(0, 0, 1201, 801))
        self.bgwidget.setStyleSheet("QWidget#bgwidget {\n"
                                    "    background-color: qlineargradient(spread:pad, x1:0.091, y1:0.101636, x2:0.991379, y2:0.977, \n"
                                    "    stop:0 rgba(0, 0, 0, 255), stop:0.95 rgba(0, 0, 255, 255), stop:1 rgba(0, 0, 255, 255));\n"
                                    "}\n"
                                    "")
        self.bgwidget.setObjectName("bgwidget")
        self.label = QtWidgets.QLabel(self.bgwidget)
        self.label.setGeometry(QtCore.QRect(530, 110, 151, 71))
        self.label.setStyleSheet("font: 36pt \"MS Shell Dlg 2\"; color:rgb(255, 255, 255)")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.bgwidget)
        self.label_2.setGeometry(QtCore.QRect(420, 200, 391, 41))
        self.label_2.setStyleSheet("font: 16pt \"MS Shell Dlg 2\";color:rgb(255, 255, 255)")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.login = QtWidgets.QPushButton(self.bgwidget)
        self.login.setGeometry(QtCore.QRect(440, 490, 341, 51))
        self.login.setStyleSheet("border-radius:20px;\n"
                                 "background-color: rgb(170, 255, 255);\n"
                                 "font: 14pt \"MS Shell Dlg 2\";")
        self.login.setObjectName("login")
        self.emailfield = QtWidgets.QLineEdit(self.bgwidget)
        self.emailfield.setGeometry(QtCore.QRect(440, 290, 341, 51))
        self.emailfield.setStyleSheet("background-color:rgba(0,0,0,0);\n"
                                      "font: 12pt \"MS Shell Dlg 2\";\n"
                                      "color: white;")
        self.emailfield.setObjectName("emailfield")
        self.passwordfield = QtWidgets.QLineEdit(self.bgwidget)
        self.passwordfield.setGeometry(QtCore.QRect(440, 390, 341, 51))
        self.passwordfield.setStyleSheet("background-color:rgba(0,0,0,0);\n"
                                         "font: 12pt \"MS Shell Dlg 2\";\n"
                                         "color: white;")
        self.passwordfield.setEchoMode(QtWidgets.QLineEdit.Password)  # Hide password characters
        self.passwordfield.setObjectName("passwordfield")
        self.label_3 = QtWidgets.QLabel(self.bgwidget)
        self.label_3.setGeometry(QtCore.QRect(440, 270, 81, 20))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(10)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("font: 10pt \"MS Shell Dlg 2\";\n"
                                   "Color: White;")
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.bgwidget)
        self.label_4.setGeometry(QtCore.QRect(440, 370, 81, 20))
        self.label_4.setStyleSheet("font: 10pt \"MS Shell Dlg 2\";\n"
                                   "Color: White;")
        self.label_4.setObjectName("label_4")
        self.error = QtWidgets.QLabel(self.bgwidget)
        self.error.setGeometry(QtCore.QRect(440, 456, 341, 20))
        self.error.setStyleSheet("font: 12pt \"MS Shell Dlg 2\"; color:red;")
        self.error.setText("")
        self.error.setObjectName("error")

        self.retranslateUi(Dialog)
        self.login.clicked.connect(self.handlelogin)  # Connect the login button to the handle_login method
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "Login"))
        self.label_2.setText(_translate("Dialog", "Sign in to your existing account"))
        self.login.setText(_translate("Dialog", "Log in"))
        self.label_3.setText(_translate("Dialog", "Username"))
        self.label_4.setText(_translate("Dialog", "Password"))

    def handlelogin(self):
        username = self.emailfield.text()
        password = self.passwordfield.text()
        if login(username, password):
            self.open_main_screen()
            self.bgwidget.parent().close()
        else:
            self.error.setText("Invalid username or password.")

    def open_main_screen(self):
        self.main_dialog = QtWidgets.QDialog()
        self.ui = Ui_Dialog_main_screen()
        self.ui.setupUi(self.main_dialog)
        self.main_dialog.show()  # Changed from exec_() to show()
        self.bgwidget.parent().close()


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1200, 800)
        self.bgwidget = QtWidgets.QWidget(Dialog)
        self.bgwidget.setGeometry(QtCore.QRect(0, 0, 1201, 801))
        self.bgwidget.setStyleSheet("QWidget#bgwidget {\n"
                                    "    background-color: qlineargradient(spread:pad, x1:0.091, y1:0.101636, x2:0.991379, y2:0.977, \n"
                                    "    stop:0 rgba(0, 0, 0, 255), stop:0.95 rgba(0, 0, 255, 255), stop:1 rgba(0, 0, 255, 255));\n"
                                    "}\n"
                                    "")
        self.bgwidget.setObjectName("bgwidget")
        self.label = QtWidgets.QLabel(self.bgwidget)
        self.label.setGeometry(QtCore.QRect(490, 230, 251, 61))
        self.label.setStyleSheet("font: 36pt \"MS Shell Dlg 2\"; color:rgb(255, 255, 255)")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.bgwidget)
        self.label_2.setGeometry(QtCore.QRect(410, 310, 391, 41))
        self.label_2.setStyleSheet("font: 16pt \"MS Shell Dlg 2\";color:rgb(255, 255, 255)")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.login = QtWidgets.QPushButton(self.bgwidget)
        self.login.setGeometry(QtCore.QRect(440, 390, 341, 51))
        self.login.setStyleSheet("border-radius:20px;\n"
                                 "background-color: rgb(170, 255, 255);\n"
                                 "font: 14pt \"MS Shell Dlg 2\";")
        self.login.setObjectName("login")
        self.create = QtWidgets.QPushButton(self.bgwidget)
        self.create.setGeometry(QtCore.QRect(440, 480, 341, 51))
        self.create.setStyleSheet("border-radius:20px;\n"
                                  "background-color: rgb(170, 255, 255);\n"
                                  "font: 14pt \"MS Shell Dlg 2\";")
        self.create.setObjectName("create")

        # Connect buttons
        self.login.clicked.connect(self.open_login_screen)  # Connect to open the login screen
        self.create.clicked.connect(self.create_account)  # You can connect this to your account creation logic

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "Welcome"))
        self.label_2.setText(_translate("Dialog", "You have successfully logged in!"))
        self.login.setText(_translate("Dialog", "Login"))
        self.create.setText(_translate("Dialog", "Create a new account"))

    def open_login_screen(self):
        self.login_dialog = QtWidgets.QDialog()
        self.ui = Ui_Dialog1()
        self.ui.setupUi(self.login_dialog)
        self.login_dialog.show()  # Use show() instead of exec_()
        self.bgwidget.parent().close()

    def create_account(self):
        self.login_dialog = QtWidgets.QDialog()
        self.ui = Ui_Dialog_signup()
        self.ui.setupUi(self.login_dialog)
        self.login_dialog.show()  # Use show() instead of exec_()
        self.bgwidget.parent().close()


class Ui_Dialog_quadratic(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1200, 799)
        self.bgwidget = QtWidgets.QWidget(Dialog)
        self.bgwidget.setGeometry(QtCore.QRect(0, 0, 1201, 801))
        self.bgwidget.setStyleSheet("QWidget#bgwidget {\n"
                                    "    background-color: qlineargradient(spread:pad, x1:0.091, y1:0.101636, x2:0.991379, y2:0.977, \n"
                                    "    stop:0 rgba(0, 0, 0, 255), stop:0.95 rgba(0, 0, 255, 255), stop:1 rgba(0, 0, 255, 255));\n"
                                    "}\n"
                                    "")
        self.bgwidget.setObjectName("bgwidget")
        self.bgwidget_2 = QtWidgets.QWidget(self.bgwidget)
        self.bgwidget_2.setGeometry(QtCore.QRect(0, 0, 1201, 801))
        self.bgwidget_2.setStyleSheet("QWidget#bgwidget {\n"
                                      "    background-color: qlineargradient(spread:pad, x1:0.091, y1:0.101636, x2:0.991379, y2:0.977, \n"
                                      "    stop:0 rgba(0, 0, 0, 255), stop:0.95 rgba(0, 0, 255, 255), stop:1 rgba(0, 0, 255, 255));\n"
                                      "}\n"
                                      "")
        self.bgwidget_2.setObjectName("bgwidget_2")
        self.label = QtWidgets.QLabel(self.bgwidget_2)
        self.label.setGeometry(QtCore.QRect(400, 110, 651, 71))
        self.label.setStyleSheet("font: 36pt \"MS Shell Dlg 2\"; color:rgb(255, 255, 255)")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.bgwidget_2)
        self.label_2.setGeometry(QtCore.QRect(420, 210, 611, 41))
        self.label_2.setStyleSheet("font: 16pt \"MS Shell Dlg 2\";color:rgb(255, 255, 255)")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.signup_3 = QtWidgets.QPushButton(self.bgwidget_2)
        self.signup_3.setGeometry(QtCore.QRect(550, 360, 341, 61))
        self.signup_3.setStyleSheet("border-radius:20px;\n"
                                    "background-color: rgb(170, 255, 255);\n"
                                    "font: 14pt \"MS Shell Dlg 2\";")
        self.signup_3.setObjectName("signup_3")
        self.signup_3.clicked.connect(self.solve_and_display)
        self.create_input_bar(280)
        self.widget = QtWidgets.QWidget(self.bgwidget_2)
        self.widget.setGeometry(QtCore.QRect(0, 0, 281, 801))
        self.widget.setStyleSheet("background-color: rgba(255, 255, 255, 50);\n"
                                  "")
        self.widget.setObjectName("widget")
        self.signup_4 = QtWidgets.QPushButton(self.widget)
        self.signup_4.setGeometry(QtCore.QRect(10, 30, 261, 51))
        self.signup_4.setStyleSheet("border-radius: 20px;\n"
                                    "color: white;\n"
                                    "font: 14pt \"MS Shell Dlg 2\";\n"
                                    "background-color: rgba(255, 255, 255, 150);\n"
                                    "")
        self.signup_4.setObjectName("signup_4")
        self.signup_5 = QtWidgets.QPushButton(self.widget)
        self.signup_5.setGeometry(QtCore.QRect(10, 100, 261, 51))
        self.signup_5.setStyleSheet("border-radius: 20px;\n"
                                    "color: white;\n"
                                    "font: 14pt \"MS Shell Dlg 2\";\n"
                                    "background-color: rgba(255, 255, 255, 150);\n"
                                    "")
        self.signup_5.setObjectName("signup_5")
        self.signup_6 = QtWidgets.QPushButton(self.widget)
        self.signup_6.setGeometry(QtCore.QRect(10, 170, 261, 51))
        self.signup_6.setStyleSheet("border-radius: 20px;\n"
                                    "color: white;\n"
                                    "font: 14pt \"MS Shell Dlg 2\";\n"
                                    "background-color: rgba(255, 255, 255, 150);\n"
                                    "")
        self.signup_6.setObjectName("signup_6")
        self.signup_7 = QtWidgets.QPushButton(self.widget)
        self.signup_7.setGeometry(QtCore.QRect(10, 730, 261, 51))
        self.signup_7.setStyleSheet("border-radius: 20px;\n"
                                    "color: Red;\n"
                                    "font: 14pt \"MS Shell Dlg 2\";\n"
                                    "background-color: rgba(255, 255, 255, 150);\n"
                                    "")
        self.signup_7.setObjectName("signup_7")
        self.signup_7.clicked.connect(QtWidgets.QApplication.quit)
        self.retranslateUi(Dialog)
        self.signup_4.clicked.connect(self.opendashboard)
        self.signup_5.clicked.connect(self.openhistory)
        self.signup_6.clicked.connect(self.openchange)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "Solve Quadratic Equations"))
        self.label_2.setText(_translate("Dialog", "Enter an Equation or a set of Equations to continue!"))
        self.signup_3.setText(_translate("Dialog", "Generate the result"))
        self.signup_4.setText(_translate("Dialog", "Dashboard"))
        self.signup_5.setText(_translate("Dialog", "Equations History"))
        self.signup_6.setText(_translate("Dialog", "Change Password"))
        self.signup_7.setText(_translate("Dialog", "Quit Application"))

    def create_input_bar(self, y_position):
        emailfield = QtWidgets.QLineEdit(self.bgwidget_2)
        emailfield.setGeometry(QtCore.QRect(550, y_position, 301, 51))
        emailfield.setStyleSheet("background-color:rgba(0,0,0,0);\n"
                                 "color: White;\n"
                                 "font: 12pt \"MS Shell Dlg 2\";")
        emailfield.setObjectName(f"emailfield_{y_position}")
        emailfield.show()
        plus_button = QtWidgets.QPushButton(self.bgwidget_2)
        plus_button.setGeometry(QtCore.QRect(860, y_position, 31, 51))
        plus_button.setStyleSheet("border-radius: 10px;\n"
                                  "background-color: rgb(170, 255, 255);\n"
                                  "font: 14pt \"MS Shell Dlg 2\";")
        plus_button.setText("+")
        plus_button.setObjectName(f"plus_button_{y_position}")
        plus_button.show()
        self.signup_3.setGeometry(QtCore.QRect(550, y_position + 80, 341, 61))
        plus_button.clicked.connect(lambda: self.add_new_input_bar())

    def add_new_input_bar(self):
        # Calculate new position for the input bar
        last_y_position = int(self.signup_3.geometry().y()) - 80
        new_y_position = last_y_position + 80
        self.create_input_bar(new_y_position)

    def solve_and_display(self):
        equations_list = []
        for child in self.bgwidget_2.children():
            if isinstance(child, QtWidgets.QLineEdit):
                equations_list.append(child.text())

        if equations_list:
            try:
                solution = solve_quadratic1(equations_list, userid)
                self.show_popup("Solution", solution)
            except Exception as e:
                self.show_popup("Error", f"Error: {str(e)}")

    def show_popup(self, title, message):
        msg_box = QtWidgets.QMessageBox()
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setIcon(QtWidgets.QMessageBox.Information if title == "Solution" else QtWidgets.QMessageBox.Critical)
        msg_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg_box.exec_()
    def opendashboard(self):
        self.main_dialog = QtWidgets.QDialog()
        self.ui = Ui_Dialog_main_screen()
        self.ui.setupUi(self.main_dialog)
        self.main_dialog.show()
        self.bgwidget.parent().close()

    def openhistory(self):
        self.main_dialog = QtWidgets.QDialog()
        self.ui = Ui_Dialog_History()
        self.ui.setupUi(self.main_dialog)
        self.main_dialog.show()
        self.bgwidget.parent().close()

    def openchange(self):
        self.main_dialog = QtWidgets.QDialog()
        self.ui = Ui_Dialog_change_password()
        self.ui.setupUi(self.main_dialog)
        self.main_dialog.show()
        self.bgwidget.parent().close()


class Ui_Dialog_linear(object):
    num_last = 1

    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1200, 799)
        self.bgwidget = QtWidgets.QWidget(Dialog)
        self.bgwidget.setGeometry(QtCore.QRect(0, 0, 1201, 801))
        self.bgwidget.setStyleSheet("QWidget#bgwidget {\n"
                                    "    background-color: qlineargradient(spread:pad, x1:0.091, y1:0.101636, x2:0.991379, y2:0.977, \n"
                                    "    stop:0 rgba(0, 0, 0, 255), stop:0.95 rgba(0, 0, 255, 255), stop:1 rgba(0, 0, 255, 255));\n"
                                    "}\n"
                                    "")
        self.bgwidget.setObjectName("bgwidget")
        self.bgwidget_2 = QtWidgets.QWidget(self.bgwidget)
        self.bgwidget_2.setGeometry(QtCore.QRect(0, 0, 1201, 801))
        self.bgwidget_2.setStyleSheet("QWidget#bgwidget {\n"
                                      "    background-color: qlineargradient(spread:pad, x1:0.091, y1:0.101636, x2:0.991379, y2:0.977, \n"
                                      "    stop:0 rgba(0, 0, 0, 255), stop:0.95 rgba(0, 0, 255, 255), stop:1 rgba(0, 0, 255, 255));\n"
                                      "}\n"
                                      "")
        self.bgwidget_2.setObjectName("bgwidget_2")
        self.label = QtWidgets.QLabel(self.bgwidget_2)
        self.label.setGeometry(QtCore.QRect(400, 110, 651, 71))
        self.label.setStyleSheet("font: 36pt \"MS Shell Dlg 2\"; color:rgb(255, 255, 255)")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.bgwidget_2)
        self.label_2.setGeometry(QtCore.QRect(420, 210, 611, 41))
        self.label_2.setStyleSheet("font: 16pt \"MS Shell Dlg 2\";color:rgb(255, 255, 255)")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.signup_3 = QtWidgets.QPushButton(self.bgwidget_2)
        self.signup_3.setGeometry(QtCore.QRect(550, 360, 341, 61))
        self.signup_3.setStyleSheet("border-radius:20px;\n"
                                    "background-color: rgb(170, 255, 255);\n"
                                    "font: 14pt \"MS Shell Dlg 2\";")
        self.signup_3.setObjectName("signup_3")
        self.signup_3.clicked.connect(self.solve_and_display)
        self.create_input_bar1(280)
        self.widget = QtWidgets.QWidget(self.bgwidget_2)
        self.widget.setGeometry(QtCore.QRect(0, 0, 281, 801))
        self.widget.setStyleSheet("background-color: rgba(255, 255, 255, 50);\n"
                                  "")
        self.widget.setObjectName("widget")
        self.signup_4 = QtWidgets.QPushButton(self.widget)
        self.signup_4.setGeometry(QtCore.QRect(10, 30, 261, 51))
        self.signup_4.setStyleSheet("border-radius: 20px;\n"
                                    "color: white;\n"
                                    "font: 14pt \"MS Shell Dlg 2\";\n"
                                    "background-color: rgba(255, 255, 255, 150);\n"
                                    "")
        self.signup_4.setObjectName("signup_4")
        self.signup_5 = QtWidgets.QPushButton(self.widget)
        self.signup_5.setGeometry(QtCore.QRect(10, 100, 261, 51))
        self.signup_5.setStyleSheet("border-radius: 20px;\n"
                                    "color: white;\n"
                                    "font: 14pt \"MS Shell Dlg 2\";\n"
                                    "background-color: rgba(255, 255, 255, 150);\n"
                                    "")
        self.signup_5.setObjectName("signup_5")
        self.signup_6 = QtWidgets.QPushButton(self.widget)
        self.signup_6.setGeometry(QtCore.QRect(10, 170, 261, 51))
        self.signup_6.setStyleSheet("border-radius: 20px;\n"
                                    "color: white;\n"
                                    "font: 14pt \"MS Shell Dlg 2\";\n"
                                    "background-color: rgba(255, 255, 255, 150);\n"
                                    "")
        self.signup_6.setObjectName("signup_6")
        self.signup_7 = QtWidgets.QPushButton(self.widget)
        self.signup_7.setGeometry(QtCore.QRect(10, 730, 261, 51))
        self.signup_7.setStyleSheet("border-radius: 20px;\n"
                                    "color: Red;\n"
                                    "font: 14pt \"MS Shell Dlg 2\";\n"
                                    "background-color: rgba(255, 255, 255, 150);\n"
                                    "")
        self.signup_7.setObjectName("signup_7")
        self.signup_7.clicked.connect(QtWidgets.QApplication.quit)
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        self.signup_4.clicked.connect(self.opendashboard)
        self.signup_5.clicked.connect(self.openhistory)
        self.signup_6.clicked.connect(self.openchange)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "Solve Linear Equations"))
        self.label_2.setText(_translate("Dialog", "Enter a linear equation or a set of linear equations to continue!"))
        self.signup_3.setText(_translate("Dialog", "Generate the result"))
        self.signup_4.setText(_translate("Dialog", "Dashboard"))
        self.signup_5.setText(_translate("Dialog", "Equations History"))
        self.signup_6.setText(_translate("Dialog", "Change Password"))
        self.signup_7.setText(_translate("Dialog", "Quit Application"))

    def create_input_bar1(self, y_position):
        emailfield = QLineEdit(self.bgwidget_2)
        emailfield.setGeometry(QRect(550, y_position, 341, 61))
        emailfield.setStyleSheet("background-color:rgba(0,0,0,0);\n"
                                 "color: White;\n"
                                 "font: 12pt \"MS Shell Dlg 2\";")
        emailfield.setObjectName(f"emailfield_{y_position}")
        emailfield.show()
        emailfield.textChanged.connect(lambda: self.check_equation(emailfield))
        self.signup_3.setGeometry(QRect(550, y_position + 80, 341, 61))
    last_y_position = 280
    def create_input_bar(self, y_position):
        emailfield = QLineEdit(self.bgwidget_2)
        emailfield.setGeometry(QRect(550, y_position, 341, 61))
        emailfield.setStyleSheet("background-color:rgba(0,0,0,0);\n"
                                 "color: White;\n"
                                 "font: 12pt \"MS Shell Dlg 2\";")
        emailfield.setObjectName(f"emailfield_{y_position}")
        emailfield.show()
        self.signup_3.setGeometry(QRect(550, y_position + 80, 341, 61))

    def check_equation(self, input_field):
        equation = input_field.text()
        variables = self.parse_equation(equation)
        if len(variables) > self.num_last:
            new_y_position = self.last_y_position + 80
            self.last_y_position=new_y_position
            self.create_input_bar(new_y_position)
            self.num_last += 1

    def parse_equation(self, equation):
        # Simple parser to extract variables from the equation
        # This assumes the equation is in the format "ax + by + cz = d"
        variables = []
        for char in equation:
            if char.isalpha():
                variables.append(char)
        return variables

    def solve_and_display(self):
        equations_list = []
        all_fields_filled = True

        for child in self.bgwidget_2.children():
            if isinstance(child, QtWidgets.QLineEdit):
                text = child.text()
                if not text.strip():  # Check if the input field is empty or contains only whitespace
                    all_fields_filled = False
                equations_list.append(text)

        if not all_fields_filled:
            self.show_popup("Error", "All fields should be filled!")
            return

        if equations_list:
            try:
                # Call the global function to solve the equations
                solution = solve_linear(equations_list, userid)
                self.show_popup("Solution", solution)
            except Exception as e:
                self.show_popup("Error", f"Error: {str(e)}")

    def show_popup(self, title, message):
        msg_box = QtWidgets.QMessageBox()
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setIcon(QtWidgets.QMessageBox.Information if title == "Solution" else QtWidgets.QMessageBox.Critical)
        msg_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg_box.exec_()

    def opendashboard(self):
        self.main_dialog = QtWidgets.QDialog()
        self.ui = Ui_Dialog_main_screen()
        self.ui.setupUi(self.main_dialog)
        self.main_dialog.show()
        self.bgwidget.parent().close()

    def openhistory(self):
        self.main_dialog = QtWidgets.QDialog()
        self.ui = Ui_Dialog_History()
        self.ui.setupUi(self.main_dialog)
        self.main_dialog.show()
        self.bgwidget.parent().close()

    def openchange(self):
        self.main_dialog = QtWidgets.QDialog()
        self.ui = Ui_Dialog_change_password()
        self.ui.setupUi(self.main_dialog)
        self.main_dialog.show()
        self.bgwidget.parent().close()


class Ui_Dialog_signup(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1200, 800)
        self.bgwidget = QtWidgets.QWidget(Dialog)
        self.bgwidget.setGeometry(QtCore.QRect(0, 0, 1201, 801))
        self.bgwidget.setStyleSheet("QWidget#bgwidget {\n"
                                    "    background-color: qlineargradient(spread:pad, x1:0.091, y1:0.101636, x2:0.991379, y2:0.977, \n"
                                    "    stop:0 rgba(0, 0, 0, 255), stop:0.95 rgba(0, 0, 255, 255), stop:1 rgba(0, 0, 255, 255));\n"
                                    "}\n"
                                    "")
        self.bgwidget.setObjectName("bgwidget")
        self.label = QtWidgets.QLabel(self.bgwidget)
        self.label.setGeometry(QtCore.QRect(510, 110, 211, 71))
        self.label.setStyleSheet("font: 36pt \"MS Shell Dlg 2\"; color:rgb(255, 255, 255)")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.bgwidget)
        self.label_2.setGeometry(QtCore.QRect(430, 200, 381, 41))
        self.label_2.setStyleSheet("font: 16pt \"MS Shell Dlg 2\";color:rgb(255, 255, 255)")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.signup = QtWidgets.QPushButton(self.bgwidget)
        self.signup.setGeometry(QtCore.QRect(440, 570, 341, 51))
        self.signup.setStyleSheet("border-radius:20px;\n"
                                  "background-color: rgb(170, 255, 255);\n"
                                  "font: 14pt \"MS Shell Dlg 2\";")
        self.signup.setObjectName("signup")
        self.emailfield = QtWidgets.QLineEdit(self.bgwidget)
        self.emailfield.setGeometry(QtCore.QRect(440, 290, 341, 51))
        self.emailfield.setStyleSheet("background-color:rgba(0,0,0,0);\n"
                                      "font: 12pt \"MS Shell Dlg 2\";\n"
                                      "color: white;")
        self.emailfield.setObjectName("emailfield")
        self.passwordfield = QtWidgets.QLineEdit(self.bgwidget)
        self.passwordfield.setGeometry(QtCore.QRect(440, 390, 341, 51))
        self.passwordfield.setStyleSheet("background-color:rgba(0,0,0,0);\n"
                                         "font: 12pt \"MS Shell Dlg 2\";\n"
                                         "color: white;")
        self.passwordfield.setEchoMode(QtWidgets.QLineEdit.Password)  # Hide the password text
        self.passwordfield.setObjectName("passwordfield")
        self.label_3 = QtWidgets.QLabel(self.bgwidget)
        self.label_3.setGeometry(QtCore.QRect(440, 270, 81, 20))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("font: 10pt \"MS Shell Dlg 2\";\n"
                                   "Color: White")
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.bgwidget)
        self.label_4.setGeometry(QtCore.QRect(440, 370, 81, 20))
        self.label_4.setStyleSheet("font: 10pt \"MS Shell Dlg 2\";\n"
                                   "Color: White;")
        self.label_4.setObjectName("label_4")
        self.error = QtWidgets.QLabel(self.bgwidget)
        self.error.setGeometry(QtCore.QRect(440, 540, 341, 20))
        self.error.setStyleSheet("font: 12pt \"MS Shell Dlg 2\"; color:red;")
        self.error.setText("")
        self.error.setObjectName("error")
        self.label_5 = QtWidgets.QLabel(self.bgwidget)
        self.label_5.setGeometry(QtCore.QRect(440, 470, 141, 20))
        self.label_5.setStyleSheet("font: 10pt \"MS Shell Dlg 2\";\n"
                                   "Color: White;")
        self.label_5.setObjectName("label_5")
        self.confirmpasswordfield = QtWidgets.QLineEdit(self.bgwidget)
        self.confirmpasswordfield.setGeometry(QtCore.QRect(440, 490, 341, 51))
        self.confirmpasswordfield.setStyleSheet("background-color:rgba(0,0,0,0);\n"
                                                "font: 12pt \"MS Shell Dlg 2\";\n"
                                                "color: white;")
        self.confirmpasswordfield.setEchoMode(QtWidgets.QLineEdit.Password)  # Hide the confirm password text
        self.confirmpasswordfield.setObjectName("confirmpasswordfield")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

        # Connect the signup button to the signup function
        self.signup.clicked.connect(self.handle_signup)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "Sign up"))
        self.label_2.setText(_translate("Dialog", "Register a new account"))
        self.signup.setText(_translate("Dialog", "Sign up"))
        self.label_3.setText(_translate("Dialog", "Username"))
        self.label_4.setText(_translate("Dialog", "Password"))
        self.label_5.setText(_translate("Dialog", "Confirm Password"))

    def handle_signup(self):
        username = self.emailfield.text()
        password = self.passwordfield.text()
        confirm_password = self.confirmpasswordfield.text()
        cur.execute('use equation_solver;')
        cur.execute('select username from userid where username=%s;', (username,))
        existing_user = cur.fetchone()
        if existing_user:
            self.error.setText("Username already taken")
        elif password != confirm_password:
            self.error.setText("Passwords do not match")
        else:
            self.error.setText("")
            insert_function(username, password)
            self.open_main_screen()

    def open_main_screen(self):
        self.main_dialog = QtWidgets.QDialog()
        self.ui = Ui_Dialog_main_screen()
        self.ui.setupUi(self.main_dialog)
        self.main_dialog.show()
        self.bgwidget.parent().close()


class Ui_Dialog_History(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1199, 803)
        self.bgwidget = QtWidgets.QWidget(Dialog)
        self.bgwidget.setGeometry(QtCore.QRect(0, 0, 1201, 801))
        self.bgwidget.setStyleSheet("QWidget#bgwidget {\n"
                                    "    background-color: qlineargradient(spread:pad, x1:0.091, y1:0.101636, x2:0.991379, y2:0.977, \n"
                                    "    stop:0 rgba(0, 0, 0, 255), stop:0.95 rgba(0, 0, 255, 255), stop:1 rgba(0, 0, 255, 255));\n"
                                    "}\n"
                                    "")
        self.bgwidget.setObjectName("bgwidget")
        self.bgwidget_2 = QtWidgets.QWidget(self.bgwidget)
        self.bgwidget_2.setGeometry(QtCore.QRect(0, 0, 1201, 801))
        self.bgwidget_2.setStyleSheet("QWidget#bgwidget {\n"
                                      "    background-color: qlineargradient(spread:pad, x1:0.091, y1:0.101636, x2:0.991379, y2:0.977, \n"
                                      "    stop:0 rgba(0, 0, 0, 255), stop:0.95 rgba(0, 0, 255, 255), stop:1 rgba(0, 0, 255, 255));\n"
                                      "}\n"
                                      "")
        self.bgwidget_2.setObjectName("bgwidget_2")
        self.label = QtWidgets.QLabel(self.bgwidget_2)
        self.label.setGeometry(QtCore.QRect(400, 110, 651, 71))
        self.label.setStyleSheet("font: 36pt \"MS Shell Dlg 2\"; color:rgb(255, 255, 255)")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.bgwidget_2)
        self.label_2.setGeometry(QtCore.QRect(420, 210, 611, 41))
        self.label_2.setStyleSheet("font: 16pt \"MS Shell Dlg 2\";color:rgb(255, 255, 255)")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.widget = QtWidgets.QWidget(self.bgwidget_2)
        self.widget.setGeometry(QtCore.QRect(0, 0, 281, 801))
        self.widget.setStyleSheet("background-color: rgba(255, 255, 255, 50);\n"
                                  "")
        self.widget.setObjectName("widget")
        self.signup_4 = QtWidgets.QPushButton(self.widget)
        self.signup_4.setGeometry(QtCore.QRect(10, 30, 261, 51))
        self.signup_4.setStyleSheet("border-radius: 20px;\n"
                                    "color: white;\n"
                                    "font: 14pt \"MS Shell Dlg 2\";\n"
                                    "background-color: rgba(255, 255, 255, 150);\n"
                                    "")
        self.signup_4.setObjectName("signup_4")
        self.signup_5 = QtWidgets.QPushButton(self.widget)
        self.signup_5.setGeometry(QtCore.QRect(10, 100, 261, 51))
        self.signup_5.setStyleSheet("border-radius: 20px;\n"
                                    "color: Black;\n"
                                    "font: 14pt \"MS Shell Dlg 2\";\n"
                                    "background-color: White;\n"
                                    "")
        self.signup_5.setObjectName("signup_5")
        self.signup_6 = QtWidgets.QPushButton(self.widget)
        self.signup_6.setGeometry(QtCore.QRect(10, 170, 261, 51))
        self.signup_6.setStyleSheet("border-radius: 20px;\n"
                                    "color: white;\n"
                                    "font: 14pt \"MS Shell Dlg 2\";\n"
                                    "background-color: rgba(255, 255, 255, 150);\n"
                                    "")
        self.signup_6.setObjectName("signup_6")
        self.signup_7 = QtWidgets.QPushButton(self.widget)
        self.signup_7.setGeometry(QtCore.QRect(10, 730, 261, 51))
        self.signup_7.setStyleSheet("border-radius: 20px;\n"
                                    "color: Red;\n"
                                    "font: 14pt \"MS Shell Dlg 2\";\n"
                                    "background-color: rgba(255, 255, 255, 150);\n"
                                    "")
        self.signup_7.setObjectName("signup_7")
        self.label_3 = QtWidgets.QLabel(self.bgwidget_2)
        self.label_3.setGeometry(QtCore.QRect(430, 430, 611, 41))
        self.label_3.setText("")
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.tableWidget = QtWidgets.QTableWidget(self.bgwidget_2)
        self.tableWidget.setGeometry(QtCore.QRect(310, 270, 871, 501))
        self.tableWidget.setStyleSheet("background-color: white;\n"  # Set table background to white
                                       "color: black;\n"  # Set text color to black
                                       "font: 14pt \"MS Shell Dlg 2\";")
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)

        # Set header styles
        header = self.tableWidget.horizontalHeader()
        header.setStyleSheet("background-color: white; color: black; border-bottom: 1px solid black;")
        header.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

        vertical_header = self.tableWidget.verticalHeader()
        vertical_header.setStyleSheet("background-color: white; color: black; border-right: 1px solid black;")

        # Set partition lines color to black
        self.tableWidget.setShowGrid(True)
        self.tableWidget.setStyleSheet(
            "QTableWidget::item:selected { background-color: lightgray; }")  # Highlight selected items
        self.tableWidget.setGridStyle(QtCore.Qt.SolidLine)
        self.tableWidget.viewport().setStyleSheet("background-color: white; border: 1px solid black;")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        self.signup_4.clicked.connect(self.opendashboard)
        self.signup_6.clicked.connect(self.openchange)
        self.signup_7.clicked.connect(QtWidgets.QApplication.quit)

        # Populate table with data
        self.populateTable(self.read(userid))

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "Equation History"))
        self.label_2.setText(_translate("Dialog", "Your previously solved equations"))
        self.signup_4.setText(_translate("Dialog", "Dashboard"))
        self.signup_5.setText(_translate("Dialog", "Equations History"))
        self.signup_6.setText(_translate("Dialog", "Change Password"))
        self.signup_7.setText(_translate("Dialog", "Quit Application"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("Dialog", "Equations"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("Dialog", "Solutions"))

    def read(self, uid):
        cur.execute(f'select history, answer from history where useride={uid};')
        data = cur.fetchone()
        equations_data = data[0] + '#'
        answer_data = data[1] + '#'

        equations = []
        answers = []
        equation = ''
        answer = ''

        for i in equations_data:
            if i == '#':
                if equation:
                    equations.append(equation)
                    equation = ''
                continue
            elif i == '$':
                if equation:
                    equations.append(equation)
                equations.append('')
                equation = ''
            equation += i

        if equation:
            equations.append(equation)

        for i in answer_data:
            if i == '#':
                if answer:  # Avoid adding empty strings
                    answers.append(answer)
                    answer = ''
                continue
            elif i == '$':  # Add new line for answers
                if answer:
                    answers.append(answer)
                answers.append('')  # Add empty string for the new line
                answer = ''
            answer += i

        if answer:  # Add the last answer if not empty
            answers.append(answer)

        return list(zip(equations, answers))

    def populateTable(self, data):
        self.tableWidget.setRowCount(len(data))
        for row, (equation, solution) in enumerate(data):
            # Create read-only items for the table
            equation_item = QtWidgets.QTableWidgetItem(equation)
            solution_item = QtWidgets.QTableWidgetItem(solution)

            # Set items to be read-only
            equation_item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            solution_item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)

            # Set item text wrapping
            solution_item.setTextAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

            # Set the text to wrap within the cell
            solution_item.setData(QtCore.Qt.UserRole, solution)

            self.tableWidget.setItem(row, 0, equation_item)
            self.tableWidget.setItem(row, 1, solution_item)
        self.tableWidget.setColumnWidth(1, 300)
        self.tableWidget.resizeColumnToContents(1)
        desired_row_height = 100  # Adjust this value as needed
        for row_index in range(self.tableWidget.rowCount()):
            self.tableWidget.setRowHeight(row_index, desired_row_height)

    def opendashboard(self):
        self.main_dialog = QtWidgets.QDialog()
        self.ui = Ui_Dialog_main_screen()
        self.ui.setupUi(self.main_dialog)
        self.main_dialog.show()
        self.bgwidget.parent().close()

    def openchange(self):
        self.main_dialog = QtWidgets.QDialog()
        self.ui = Ui_Dialog_change_password()
        self.ui.setupUi(self.main_dialog)
        self.main_dialog.show()
        self.bgwidget.parent().close()


class Ui_Dialog_main_screen(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1200, 802)
        self.bgwidget = QtWidgets.QWidget(Dialog)
        self.bgwidget.setGeometry(QtCore.QRect(0, 0, 1201, 801))
        self.bgwidget.setStyleSheet("QWidget#bgwidget {\n"
                                    "    background-color: qlineargradient(spread:pad, x1:0.091, y1:0.101636, x2:0.991379, y2:0.977, \n"
                                    "    stop:0 rgba(0, 0, 0, 255), stop:0.95 rgba(0, 0, 255, 255), stop:1 rgba(0, 0, 255, 255));\n"
                                    "}\n"
                                    "")
        self.bgwidget.setObjectName("bgwidget")
        self.bgwidget_2 = QtWidgets.QWidget(self.bgwidget)
        self.bgwidget_2.setGeometry(QtCore.QRect(0, 0, 1201, 801))
        self.bgwidget_2.setStyleSheet("QWidget#bgwidget {\n"
                                      "    background-color: qlineargradient(spread:pad, x1:0.091, y1:0.101636, x2:0.991379, y2:0.977, \n"
                                      "    stop:0 rgba(0, 0, 0, 255), stop:0.95 rgba(0, 0, 255, 255), stop:1 rgba(0, 0, 255, 255));\n"
                                      "}\n"
                                      "")
        self.bgwidget_2.setObjectName("bgwidget_2")
        self.label = QtWidgets.QLabel(self.bgwidget_2)
        self.label.setGeometry(QtCore.QRect(410, 110, 651, 71))
        self.label.setStyleSheet("font: 36pt \"MS Shell Dlg 2\"; color:rgb(255, 255, 255)")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.bgwidget_2)
        self.label_2.setGeometry(QtCore.QRect(450, 210, 571, 41))
        self.label_2.setStyleSheet("font: 16pt \"MS Shell Dlg 2\";color:rgb(255, 255, 255)")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.signup = QtWidgets.QPushButton(self.bgwidget_2)
        self.signup.setGeometry(QtCore.QRect(380, 300, 341, 141))
        self.signup.setStyleSheet("border-radius:20px;\n"
                                  "background-color: rgb(170, 255, 255);\n"
                                  "font: 14pt \"MS Shell Dlg 2\";")
        self.signup.setObjectName("signup")
        self.error = QtWidgets.QLabel(self.bgwidget_2)
        self.error.setGeometry(QtCore.QRect(440, 540, 341, 20))
        self.error.setStyleSheet("font: 12pt \"MS Shell Dlg 2\"; color:red;")
        self.error.setText("")
        self.error.setObjectName("error")
        self.signup_2 = QtWidgets.QPushButton(self.bgwidget_2)
        self.signup_2.setGeometry(QtCore.QRect(740, 300, 341, 141))
        self.signup_2.setStyleSheet("border-radius:20px;\n"
                                    "background-color: rgb(170, 255, 255);\n"
                                    "font: 14pt \"MS Shell Dlg 2\";")
        self.signup_2.setObjectName("signup_2")
        self.signup_3 = QtWidgets.QPushButton(self.bgwidget_2)
        self.signup_3.setGeometry(QtCore.QRect(560, 460, 341, 141))
        self.signup_3.setStyleSheet("border-radius:20px;\n"
                                    "background-color: rgb(170, 255, 255);\n"
                                    "font: 14pt \"MS Shell Dlg 2\";")
        self.signup_3.setObjectName("signup_3")
        self.widget = QtWidgets.QWidget(self.bgwidget_2)
        self.widget.setGeometry(QtCore.QRect(0, 0, 281, 801))
        self.widget.setStyleSheet("background-color: rgba(255, 255, 255, 50);\n"
                                  "")
        self.widget.setObjectName("widget")
        self.signup_4 = QtWidgets.QPushButton(self.widget)
        self.signup_4.setGeometry(QtCore.QRect(10, 30, 261, 51))
        self.signup_4.setStyleSheet("border-radius: 20px;\n"
                                    "color: black;\n"
                                    "font: 14pt \"MS Shell Dlg 2\";\n"
                                    "background-color: white;\n"
                                    "")
        self.signup_4.setObjectName("signup_4")
        self.signup_5 = QtWidgets.QPushButton(self.widget)
        self.signup_5.setGeometry(QtCore.QRect(10, 100, 261, 51))
        self.signup_5.setStyleSheet("border-radius: 20px;\n"
                                    "color: white;\n"
                                    "font: 14pt \"MS Shell Dlg 2\";\n"
                                    "background-color: rgba(255, 255, 255, 150);\n"
                                    "")
        self.signup_5.setObjectName("signup_5")
        self.signup_6 = QtWidgets.QPushButton(self.widget)
        self.signup_6.setGeometry(QtCore.QRect(10, 170, 261, 51))
        self.signup_6.setStyleSheet("border-radius: 20px;\n"
                                    "color: white;\n"
                                    "font: 14pt \"MS Shell Dlg 2\";\n"
                                    "background-color: rgba(255, 255, 255, 150);\n"
                                    "")
        self.signup_6.setObjectName("signup_6")
        self.signup_7 = QtWidgets.QPushButton(self.widget)
        self.signup_7.setGeometry(QtCore.QRect(10, 730, 261, 51))
        self.signup_7.setStyleSheet("border-radius: 20px;\n"
                                    "color: Red;\n"
                                    "font: 14pt \"MS Shell Dlg 2\";\n"
                                    "background-color: rgba(255, 255, 255, 150);\n"
                                    "")
        self.signup_7.setObjectName("signup_7")
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        self.signup.clicked.connect(self.openlinear)
        self.signup_2.clicked.connect(self.openquadratic)
        self.signup_3.clicked.connect(self.opencubic)
        self.signup_5.clicked.connect(self.openhistory)
        self.signup_6.clicked.connect(self.change)
        self.signup_7.clicked.connect(QtWidgets.QApplication.quit)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "Welcome Back!"))
        self.label_2.setText(_translate("Dialog", "Visualize solutions, one equation at a time."))
        self.signup.setText(_translate("Dialog", "Solve linear equations"))
        self.signup_2.setText(_translate("Dialog", "Solve Quadratic equations"))
        self.signup_3.setText(_translate("Dialog", "Solve Cubic equations"))
        self.signup_4.setText(_translate("Dialog", "Dashboard"))
        self.signup_5.setText(_translate("Dialog", "Equations History"))
        self.signup_6.setText(_translate("Dialog", "Change Password"))
        self.signup_7.setText(_translate("Dialog", "Quit Apllication"))

    def openlinear(self):
        self.main_dialog = QtWidgets.QDialog()
        self.ui = Ui_Dialog_linear()
        self.ui.setupUi(self.main_dialog)
        self.main_dialog.show()
        self.bgwidget.parent().close()

    def openquadratic(self):
        self.main_dialog = QtWidgets.QDialog()
        self.ui = Ui_Dialog_quadratic()
        self.ui.setupUi(self.main_dialog)
        self.main_dialog.show()
        self.bgwidget.parent().close()

    def opencubic(self):
        self.main_dialog = QtWidgets.QDialog()
        self.ui = Ui_Dialog_cubic()
        self.ui.setupUi(self.main_dialog)
        self.main_dialog.show()
        self.bgwidget.parent().close()

    def openhistory(self):
        self.main_dialog = QtWidgets.QDialog()
        self.ui = Ui_Dialog_History()
        self.ui.setupUi(self.main_dialog)
        self.main_dialog.show()
        self.bgwidget.parent().close()

    def change(self):
        self.main_dialog = QtWidgets.QDialog()
        self.ui = Ui_Dialog_change_password()
        self.ui.setupUi(self.main_dialog)
        self.main_dialog.show()
        self.bgwidget.parent().close()


class Ui_Dialog_cubic(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1200, 799)
        self.bgwidget = QtWidgets.QWidget(Dialog)
        self.bgwidget.setGeometry(QtCore.QRect(0, 0, 1201, 801))
        self.bgwidget.setStyleSheet("QWidget#bgwidget {\n"
                                    "    background-color: qlineargradient(spread:pad, x1:0.091, y1:0.101636, x2:0.991379, y2:0.977, \n"
                                    "    stop:0 rgba(0, 0, 0, 255), stop:0.95 rgba(0, 0, 255, 255), stop:1 rgba(0, 0, 255, 255));\n"
                                    "}\n"
                                    "")
        self.bgwidget.setObjectName("bgwidget")
        self.bgwidget_2 = QtWidgets.QWidget(self.bgwidget)
        self.bgwidget_2.setGeometry(QtCore.QRect(0, 0, 1201, 801))
        self.bgwidget_2.setStyleSheet("QWidget#bgwidget {\n"
                                      "    background-color: qlineargradient(spread:pad, x1:0.091, y1:0.101636, x2:0.991379, y2:0.977, \n"
                                      "    stop:0 rgba(0, 0, 0, 255), stop:0.95 rgba(0, 0, 255, 255), stop:1 rgba(0, 0, 255, 255));\n"
                                      "}\n"
                                      "")
        self.bgwidget_2.setObjectName("bgwidget_2")
        self.label = QtWidgets.QLabel(self.bgwidget_2)
        self.label.setGeometry(QtCore.QRect(400, 110, 651, 71))
        self.label.setStyleSheet("font: 36pt \"MS Shell Dlg 2\"; color:rgb(255, 255, 255)")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.bgwidget_2)
        self.label_2.setGeometry(QtCore.QRect(420, 210, 611, 41))
        self.label_2.setStyleSheet("font: 16pt \"MS Shell Dlg 2\";color:rgb(255, 255, 255)")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.signup_3 = QtWidgets.QPushButton(self.bgwidget_2)
        self.signup_3.setGeometry(QtCore.QRect(550, 360, 341, 61))
        self.signup_3.setStyleSheet("border-radius:20px;\n"
                                    "background-color: rgb(170, 255, 255);\n"
                                    "font: 14pt \"MS Shell Dlg 2\";")
        self.signup_3.setObjectName("signup_3")
        self.signup_3.clicked.connect(self.solve_and_display)
        self.create_input_bar(280)
        self.widget = QtWidgets.QWidget(self.bgwidget_2)
        self.widget.setGeometry(QtCore.QRect(0, 0, 281, 801))
        self.widget.setStyleSheet("background-color: rgba(255, 255, 255, 50);\n"
                                  "")
        self.widget.setObjectName("widget")
        self.signup_4 = QtWidgets.QPushButton(self.widget)
        self.signup_4.setGeometry(QtCore.QRect(10, 30, 261, 51))
        self.signup_4.setStyleSheet("border-radius: 20px;\n"
                                    "color: white;\n"
                                    "font: 14pt \"MS Shell Dlg 2\";\n"
                                    "background-color: rgba(255, 255, 255, 150);\n"
                                    "")
        self.signup_4.setObjectName("signup_4")
        self.signup_5 = QtWidgets.QPushButton(self.widget)
        self.signup_5.setGeometry(QtCore.QRect(10, 100, 261, 51))
        self.signup_5.setStyleSheet("border-radius: 20px;\n"
                                    "color: white;\n"
                                    "font: 14pt \"MS Shell Dlg 2\";\n"
                                    "background-color: rgba(255, 255, 255, 150);\n"
                                    "")
        self.signup_5.setObjectName("signup_5")
        self.signup_6 = QtWidgets.QPushButton(self.widget)
        self.signup_6.setGeometry(QtCore.QRect(10, 170, 261, 51))
        self.signup_6.setStyleSheet("border-radius: 20px;\n"
                                    "color: white;\n"
                                    "font: 14pt \"MS Shell Dlg 2\";\n"
                                    "background-color: rgba(255, 255, 255, 150);\n"
                                    "")
        self.signup_6.setObjectName("signup_6")
        self.signup_7 = QtWidgets.QPushButton(self.widget)
        self.signup_7.setGeometry(QtCore.QRect(10, 730, 261, 51))
        self.signup_7.setStyleSheet("border-radius: 20px;\n"
                                    "color: Red;\n"
                                    "font: 14pt \"MS Shell Dlg 2\";\n"
                                    "background-color: rgba(255, 255, 255, 150);\n"
                                    "")
        self.signup_7.setObjectName("signup_7")
        self.signup_7.clicked.connect(QtWidgets.QApplication.quit)
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        self.signup_4.clicked.connect(self.opendashboard)
        self.signup_5.clicked.connect(self.openhistory)
        self.signup_6.clicked.connect(self.openchange)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "Solve Cubic Equations"))
        self.label_2.setText(_translate("Dialog", "Enter an Equation or a set of Equations to continue!"))
        self.signup_3.setText(_translate("Dialog", "Generate the result"))
        self.signup_4.setText(_translate("Dialog", "Dashboard"))
        self.signup_5.setText(_translate("Dialog", "Equations History"))
        self.signup_6.setText(_translate("Dialog", "Change Password"))
        self.signup_7.setText(_translate("Dialog", "Quit Application"))

    def create_input_bar(self, y_position):
        emailfield = QtWidgets.QLineEdit(self.bgwidget_2)
        emailfield.setGeometry(QtCore.QRect(550, y_position, 301, 51))
        emailfield.setStyleSheet("background-color:rgba(0,0,0,0);\n"
                                 "color: White;\n"
                                 "font: 12pt \"MS Shell Dlg 2\";")
        emailfield.setObjectName(f"emailfield_{y_position}")
        emailfield.show()
        plus_button = QtWidgets.QPushButton(self.bgwidget_2)
        plus_button.setGeometry(QtCore.QRect(860, y_position, 31, 51))
        plus_button.setStyleSheet("border-radius: 10px;\n"
                                  "background-color: rgb(170, 255, 255);\n"
                                  "font: 14pt \"MS Shell Dlg 2\";")
        plus_button.setText("+")
        plus_button.setObjectName(f"plus_button_{y_position}")
        plus_button.show()
        self.signup_3.setGeometry(QtCore.QRect(550, y_position + 80, 341, 61))
        plus_button.clicked.connect(lambda: self.add_new_input_bar())

    def add_new_input_bar(self):
        # Calculate new position for the input bar
        last_y_position = int(self.signup_3.geometry().y()) - 80
        new_y_position = last_y_position + 80
        self.create_input_bar(new_y_position)

    def solve_and_display(self):
        equations_list = []
        for child in self.bgwidget_2.children():
            if isinstance(child, QtWidgets.QLineEdit):
                equations_list.append(child.text())

        if equations_list:
            try:
                solution = solve_cubic(equations_list, userid)
                self.show_popup("Solution", solution)
            except Exception as e:
                self.show_popup("Error", f"Error: {str(e)}")

    def show_popup(self, title, message):
        msg_box = QtWidgets.QMessageBox()
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setIcon(QtWidgets.QMessageBox.Information if title == "Solution" else QtWidgets.QMessageBox.Critical)
        msg_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg_box.exec_()

    def opendashboard(self):
        self.main_dialog = QtWidgets.QDialog()
        self.ui = Ui_Dialog_main_screen()
        self.ui.setupUi(self.main_dialog)
        self.main_dialog.show()
        self.bgwidget.parent().close()

    def openhistory(self):
        self.main_dialog = QtWidgets.QDialog()
        self.ui = Ui_Dialog_History()
        self.ui.setupUi(self.main_dialog)
        self.main_dialog.show()
        self.bgwidget.parent().close()

    def openchange(self):
        self.main_dialog = QtWidgets.QDialog()
        self.ui = Ui_Dialog_change_password()
        self.ui.setupUi(self.main_dialog)
        self.main_dialog.show()
        self.bgwidget.parent().close()


class Ui_Dialog_change_password(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1200, 800)
        self.bgwidget = QtWidgets.QWidget(Dialog)
        self.bgwidget.setGeometry(QtCore.QRect(0, 0, 1201, 801))
        self.bgwidget.setStyleSheet("QWidget#bgwidget {\n"
                                    "    background-color: qlineargradient(spread:pad, x1:0.091, y1:0.101636, x2:0.991379, y2:0.977, \n"
                                    "    stop:0 rgba(0, 0, 0, 255), stop:0.95 rgba(0, 0, 255, 255), stop:1 rgba(0, 0, 255, 255));\n"
                                    "}\n")
        self.bgwidget.setObjectName("bgwidget")

        # Title and instructions
        self.label = QtWidgets.QLabel(self.bgwidget)
        self.label.setGeometry(QtCore.QRect(500, 110, 500, 71))
        self.label.setStyleSheet("font: 36pt \"MS Shell Dlg 2\"; color:rgb(255, 255, 255)")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")

        self.label_2 = QtWidgets.QLabel(self.bgwidget)
        self.label_2.setGeometry(QtCore.QRect(570, 200, 381, 41))
        self.label_2.setStyleSheet("font: 16pt \"MS Shell Dlg 2\";color:rgb(255, 255, 255)")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")

        # Old password field
        self.oldpasswordfield = QtWidgets.QLineEdit(self.bgwidget)
        self.oldpasswordfield.setGeometry(QtCore.QRect(580, 290, 341, 51))
        self.oldpasswordfield.setStyleSheet("background-color:rgba(0,0,0,0);\n"
                                            "font: 12pt \"MS Shell Dlg 2\";\n"
                                            "color: white;")
        self.oldpasswordfield.setEchoMode(QtWidgets.QLineEdit.Password)
        self.oldpasswordfield.setObjectName("oldpasswordfield")

        # New password field
        self.passwordfield = QtWidgets.QLineEdit(self.bgwidget)
        self.passwordfield.setGeometry(QtCore.QRect(580, 390, 341, 51))
        self.passwordfield.setStyleSheet("background-color:rgba(0,0,0,0);\n"
                                         "font: 12pt \"MS Shell Dlg 2\";\n"
                                         "color: white;")
        self.passwordfield.setEchoMode(QtWidgets.QLineEdit.Password)
        self.passwordfield.setObjectName("passwordfield")

        # Confirm password field
        self.confirmpasswordfield = QtWidgets.QLineEdit(self.bgwidget)
        self.confirmpasswordfield.setGeometry(QtCore.QRect(580, 490, 341, 51))
        self.confirmpasswordfield.setStyleSheet("background-color:rgba(0,0,0,0);\n"
                                                "font: 12pt \"MS Shell Dlg 2\";\n"
                                                "color: white;")
        self.confirmpasswordfield.setEchoMode(QtWidgets.QLineEdit.Password)
        self.confirmpasswordfield.setObjectName("confirmpasswordfield")

        # Labels for the fields
        self.label_3 = QtWidgets.QLabel(self.bgwidget)
        self.label_3.setGeometry(QtCore.QRect(580, 270, 81, 20))
        self.label_3.setStyleSheet("font: 10pt \"MS Shell Dlg 2\";\n"
                                   "Color: White;")
        self.label_3.setObjectName("label_3")

        self.label_4 = QtWidgets.QLabel(self.bgwidget)
        self.label_4.setGeometry(QtCore.QRect(580, 370, 81, 20))
        self.label_4.setStyleSheet("font: 10pt \"MS Shell Dlg 2\";\n"
                                   "Color: White;")
        self.label_4.setObjectName("label_4")

        self.label_5 = QtWidgets.QLabel(self.bgwidget)
        self.label_5.setGeometry(QtCore.QRect(580, 470, 141, 20))
        self.label_5.setStyleSheet("font: 10pt \"MS Shell Dlg 2\";\n"
                                   "Color: White;")
        self.label_5.setObjectName("label_5")

        # Error message label
        self.error = QtWidgets.QLabel(self.bgwidget)
        self.error.setGeometry(QtCore.QRect(580, 540, 341, 20))
        self.error.setStyleSheet("font: 12pt \"MS Shell Dlg 2\"; color:red;")
        self.error.setText("")
        self.error.setObjectName("error")

        # Change Password button
        self.changepassword = QtWidgets.QPushButton(self.bgwidget)
        self.changepassword.setGeometry(QtCore.QRect(580, 570, 341, 51))
        self.changepassword.setStyleSheet("border-radius:20px;\n"
                                          "background-color: rgb(170, 255, 255);\n"
                                          "font: 14pt \"MS Shell Dlg 2\";")
        self.changepassword.setObjectName("changepassword")

        # Sidebar buttons
        self.widget = QtWidgets.QWidget(self.bgwidget)
        self.widget.setGeometry(QtCore.QRect(0, 0, 281, 801))
        self.widget.setStyleSheet("background-color: rgba(255, 255, 255, 50);")
        self.widget.setObjectName("widget")

        self.signup_4 = QtWidgets.QPushButton(self.widget)
        self.signup_4.setGeometry(QtCore.QRect(10, 30, 261, 51))
        self.signup_4.setStyleSheet("border-radius: 20px;\n"
                                    "color: white;\n"
                                    "font: 14pt \"MS Shell Dlg 2\";\n"
                                    "background-color: rgba(255, 255, 255, 150);\n"
                                    "")
        self.signup_4.setObjectName("signup_4")

        self.signup_5 = QtWidgets.QPushButton(self.widget)
        self.signup_5.setGeometry(QtCore.QRect(10, 100, 261, 51))
        self.signup_5.setStyleSheet("border-radius: 20px;\n"
                                    "color: white;\n"
                                    "font: 14pt \"MS Shell Dlg 2\";\n"
                                    "background-color: rgba(255, 255, 255, 150);\n"
                                    "")
        self.signup_5.setObjectName("signup_5")

        self.signup_6 = QtWidgets.QPushButton(self.widget)
        self.signup_6.setGeometry(QtCore.QRect(10, 170, 261, 51))
        self.signup_6.setStyleSheet("border-radius: 20px;\n"
                                    "color: Black;\n"
                                    "font: 14pt \"MS Shell Dlg 2\";\n"
                                    "background-color: White;\n"
                                    "")
        self.signup_6.setObjectName("signup_6")

        self.signup_7 = QtWidgets.QPushButton(self.widget)
        self.signup_7.setGeometry(QtCore.QRect(10, 730, 261, 51))
        self.signup_7.setStyleSheet("border-radius: 20px;\n"
                                    "color: Red;\n"
                                    "font: 14pt \"MS Shell Dlg 2\";\n"
                                    "background-color: rgba(255, 255, 255, 150);\n"
                                    "")
        self.signup_7.setObjectName("signup_7")
        self.signup_7.clicked.connect(QtWidgets.QApplication.quit)

        # Connect signals
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        self.changepassword.clicked.connect(self.handle_change_password)
        self.signup_4.clicked.connect(self.open_main_screen)
        self.signup_5.clicked.connect(self.open_equation_history)

        self.retranslateUi(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Change Password"))
        self.label.setText(_translate("Dialog", "Change Password"))
        self.label_2.setText(_translate("Dialog", "Update your account password"))
        self.changepassword.setText(_translate("Dialog", "Change Password"))
        self.label_3.setText(_translate("Dialog", "Old Password"))
        self.label_4.setText(_translate("Dialog", "New Password"))
        self.label_5.setText(_translate("Dialog", "Confirm Password"))
        self.signup_4.setText(_translate("Dialog", "Dashboard"))
        self.signup_5.setText(_translate("Dialog", "Equations History"))
        self.signup_6.setText(_translate("Dialog", "Change Password"))
        self.signup_7.setText(_translate("Dialog", "Exit"))

    def handle_change_password(self):
        oldpassword = self.oldpasswordfield.text()
        newpassword = self.passwordfield.text()
        confirm_password = self.confirmpasswordfield.text()

        cur.execute('use equation_solver;')
        cur.execute('SELECT password FROM userid WHERE useride = 1;')
        existing_password = cur.fetchone()

        if existing_password and existing_password[0] != oldpassword:
            self.error.setText("Old password is incorrect")
        elif newpassword != confirm_password:
            self.error.setText("New passwords do not match")
        else:
            self.error.setText("")
            update_function(oldpassword, newpassword)
            self.error.setText("Password Changed!")

    def open_main_screen(self):
        self.main_dialog = QtWidgets.QDialog()
        self.ui = Ui_Dialog_main_screen()
        self.ui.setupUi(self.main_dialog)
        self.main_dialog.show()
        self.bgwidget.parent().close()

    def open_equation_history(self):
        self.main_dialog = QtWidgets.QDialog()
        self.ui = Ui_Dialog_History()
        self.ui.setupUi(self.main_dialog)
        self.main_dialog.show()
        self.bgwidget.parent().close()


create()
app = QtWidgets.QApplication(sys.argv)
Dialog = QtWidgets.QDialog()
ui = Ui_Dialog()
ui.setupUi(Dialog)
Dialog.show()
sys.exit(app.exec_())
