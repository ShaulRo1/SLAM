import pygame
from pygame.locals import *
import numpy as np
import random
from OpenGL.GL import *
from OpenGL.GLU import *
import sol3 as s3
from skimage import feature
import matplotlib.pyplot as plt
display = (400, 300)

import math
def get_cube_information():

    vertices = (
        (1, -1, -1),
        (1, 1, -1),
        (-1, 1, -1),
        (-1, -1, -1),
        (1, -1, 1),
        (1, 1, 1, ),
        (-1, -1, 1),
        (-1, 1, 1),
        )

    edges = (
        (0,1),
        (0,3),
        (0,4),
        (2,1),
        (2,3),
        (2,7),
        (6,3),
        (6,4),
        (6,7),
        (5,1),
        (5,4),
        (5,7),
        )

    surfaces = (
        (0,1,2,3),
        (3,2,7,6),
        (6,7,5,4),
        (4,5,1,0),
        (1,5,7,2),
        (4,0,3,6),
        )

    colors = (
        (1.000, 0.920, 0.000),
        (0.000, 0.860, 0.000),
        (1.000, 0.480, 0.000),
        (1.000, 1.000, 1.000),
        (0.900, 0.000, 0.000),
        (0.000, 0.000, 0.950)
    )
    return vertices, edges, surfaces, colors

def get_rand_pixel():
    h, w = display
    row = random.randint(0, h - 1)
    col = random.randint(0, w - 1)
    return row, col


def get_n_rand_pixels(n):
    pixels = [[200, 150], [190, 160], [180,120]]
    # pixels = []
    # while n > 0:
    #     x, y = get_rand_pixel()
    #     cur_pixel = [x, y]
    #     if cur_pixel in pixels:
    #         continue
    #     pixels.append(cur_pixel)
    #     n -= 1
    return pixels

def get_plane_from_points(points):
    points = np.array(points).reshape(3, 3)
    p1, p2, p3 = points
    v1 = p3 - p1
    v2 = p2 - p1
    cp = np.cross(v1, v2)
    a, b, c = cp
    d = np.dot(cp, p3)
    return a, b, c, d

def apply_plane_on_point(point, plane):
    x, y, z = point
    a, b, c, d = plane
    return a * x + b * y + c * z + d


def create_depth_image(im_shape, name):
    row, col = 0, 0
    new_im = np.zeros(im_shape)
    while row < im_shape[0]:
        while col < im_shape[1]:
            depth = glReadPixels(row, col, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)
            new_im[row, col] = depth
            col += 1
        row += 1
        col = 0
    plt.imsave(name, new_im)
    # show_images([new_im.T], 1)


def ransac_plane(num_of_iterations):
    best_plane_agreement = 0
    best_plane = (0, 0, 0, 0)
    all_ims = []
    best_im = []
    should_choose = True
    threshold = 0.1
    while num_of_iterations > 0:
        if should_choose:
            threshold = float(input('choose threshold: '))
        print('--------------'+str(num_of_iterations))
        cur_agreement = 0
        points = get_n_points_in_world(3)
        plane = get_plane_from_points(points)
        print(plane)
        row = 0
        col = 0
        cur_im = np.zeros(display)
        while row < display[0]:
            while col < display[1]:
                point = turn_pixels_to_points([[row, col]])[0][0]
                diff = apply_plane_on_point(point, plane)
                if abs(diff) < threshold:
                    cur_agreement += 1
                    cur_im[row, col] = 1
                col += 1
            col = 0
            row += 1
        print('done')
        if cur_agreement < 100:
            threshold = threshold * 1.1
            print(threshold)
            should_choose = False
            continue
        else:
            show_images([cur_im.T], 1)
            should_choose = True
            print(threshold)
            is_good = input("Is plane good?: ")
            if is_good == 'y_1':
                print(plane)
                return ('1', plane, threshold)
            if is_good == 'y_2':
                print(plane)
                return('2', plane, threshold)
        cur_im = cur_im.T
        all_ims.append(cur_im)
        if cur_agreement > best_plane_agreement:
            best_plane = plane
            best_plane_agreement = cur_agreement
            best_im = cur_im
        num_of_iterations -= 1
        print(cur_agreement)
    return best_plane, all_ims, best_im


def Cube():
    glBegin(GL_QUADS)

    (vertices, edges, surfaces, colors) = get_cube_information()
    for i, surface in enumerate(surfaces):
        x = 0
        color = colors[i]
        for vertex in surface:
            x += 1
            glColor3fv(color)
            glVertex3fv(vertices[vertex])


    glEnd()

    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])

    glEnd()


def turn_pixels_to_points(pixels):
    points = []
    for x,y in pixels:
        # get the fragment depth
        depth = glReadPixels(x, y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)
        # get projection matrix, view matrix and the viewport rectangle
        model_view = np.array(glGetDoublev(GL_MODELVIEW_MATRIX))
        proj = np.array(glGetDoublev(GL_PROJECTION_MATRIX))
        view = np.array(glGetIntegerv(GL_VIEWPORT))
        # unproject the point
        point = gluUnProject(x, y, depth, model_view, proj, view)
        points.append([point])
    return points

def get_n_points_in_world(n):
    pixels = get_n_rand_pixels(n)
    return turn_pixels_to_points(pixels)



def show_images(images, cols=1, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


def render_cube():
    pygame.init()
    tx = 0
    ty = 0
    tz = 0
    ry = 0
    rx = 0
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL | RESIZABLE)

    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)

    view_mat = np.matrix(np.identity(4), copy=False, dtype='float32')

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glTranslatef(0, 0, 0)
    glGetFloatv(GL_MODELVIEW_MATRIX, view_mat)
    glLoadIdentity()
    plane1, plane2 = None, None
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    quit()
                if event.key == pygame.K_a:
                    tx = 0.05
                elif event.key == pygame.K_d:
                    tx = -0.05
                elif event.key == pygame.K_w:
                    tz = 0.05
                elif event.key == pygame.K_s:
                    tz = -0.05
                elif event.key == pygame.K_RIGHT:
                    ry = 1.0
                elif event.key == pygame.K_LEFT:
                    ry = -1.0
                elif event.key == pygame.K_UP:
                    rx = -1.0
                elif event.key == pygame.K_DOWN:
                    rx = 1.0
                elif event.key == pygame.K_SPACE:
                    print(plane1)
                    print(plane2)
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_a and tx > 0:
                    tx = 0
                elif event.key == pygame.K_d and tx < 0:
                    tx = 0
                elif event.key == pygame.K_w and tz > 0:
                    tz = 0
                elif event.key == pygame.K_s and tz < 0:
                    tz = 0
                elif event.key == pygame.K_RIGHT and ry > 0:
                    ry = 0.0
                elif event.key == pygame.K_LEFT and ry < 0:
                    ry = 0.0
                elif event.key == pygame.K_DOWN and rx > 0:
                    rx = 0.0
                elif event.key == pygame.K_UP and rx < 0:
                    rx = 0.0
            elif event.type == pygame.MOUSEBUTTONDOWN:
                t, plane, threshold = ransac_plane(60)
                name = '/cs/usr/shaul_ro/safe/SLAM/assets/plane_'+str(t)+'_'+str(threshold)+'.png'
                if t == '1':
                    plane1 = plane
                    create_depth_image(display, name)
                if t == '2':
                    plane2 = plane
                    create_depth_image(display, name)
                # show_images(all_ims, 2)

        glPushMatrix()
        glLoadIdentity()
        glTranslatef(tx, ty, tz)
        glRotatef(ry, 0, 1, 0)
        glRotatef(rx, 1, 0, 0)
        # x, y = pygame.mouse.get_pos()
        # print(x,y)
        glMultMatrixf(view_mat)
        glGetFloatv(GL_MODELVIEW_MATRIX, view_mat)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        Cube()
        glPopMatrix()

        pygame.display.flip()
        pygame.time.wait(10)



def get_plane_n_d_rep(plane):
    a, b, c, d = plane
    norm = math.sqrt(a * a + b * b + c * c)
    n = np.divide(plane[:3], norm)
    dist =  -1 * d  / norm
    return n, dist


# def get_plane_in_image(im, plane, thresh):
#     new_im = np.zeros(im.shape)
#     h, w = im.shape
#     r, c = 0, 0
#     while r < h:
#         while c < w:
#             if apply_plane_on_point()

def main():
    im1 = s3.read_image('/cs/+/usr/shaul_ro/SLAM/assets/plane_1_0.2.png', s3.YIQ_REP).T
    im2 = s3.read_image('/cs/+/usr/shaul_ro/SLAM/assets/plane_2_0.12100000000000002.png',
                        s3.YIQ_REP).T

    show_images([im1, im2], 1)
    pixels = get_n_rand_pixels(3)
    points_1 = []
    points_2 = []
    for p in pixels:
        x, y = p
        p1 = np.array([y, x, im1[y, x]])
        p2 = np.array([y, x, im2[y, x]])
        points_1.append(p1)
        points_2.append(p2)

    plane1 = get_plane_from_points(points_1)
    plane2 = get_plane_from_points(points_2)

    thresh_1 = 0.2
    thresh_2 = 0.121


    # render_cube()
    #
    # p = (-0.010804695551434103, 2.0726593186071868e-05, -0.016621270618300255, 0.0356358028659609)
    # print(get_plane_n_d_rep(p))

    # p1 = (-0.0, 0.0, -0.020169288019536727, 0.04638929759697595)
    # p2 = (-0.012960147339368868, 3.07995631960295e-07, -0.0265530655517785, 0.07148862600900646)
    # im1  = np.zeros(display)
    # im2  = np.zeros(display)
    # row = 0
    # col = 0
    # while row < display[0]:
    #     while col < display[1]:
    #         point = turn_pixels_to_points([[row, col]])[0][0]
    #         diff = apply_plane_on_point(point, plane)
    #         if abs(diff) < threshold:
    #             cur_agreement += 1
    #             cur_im[row, col] = 1
    #         col += 1
    #     col = 0
    #     row += 1
    #     print(row)
    # show_images([cur_im.T], 1)



    # x = np.zeros((255, 255))
    # x = np.zeros((255, 255), dtype=np.uint8)
    # x[:] = np.arange(255)
    # plt.imsave('/cs/usr/shaul_ro/safe/SLAM/assets/gradient.png', x)
main()