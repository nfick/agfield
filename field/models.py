from django.db import models
from ast import literal_eval
from imageio import imread
import cv2
import numpy as np
import math
import pandas as pd
import geojson
import io
import base64
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# Create your models here.

class FindMap():
    def __init__(self, coordinates, img):
        img = img.split(',')[1]
        #print(img)
        self.img = imread(io.BytesIO(base64.b64decode(img)))
        self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)

        coordinates = literal_eval(coordinates)
        lat = coordinates['Latitude']
        lon = coordinates['Longitude']
        self.sw_lat = coordinates['SW Latitude']
        self.sw_lon = coordinates['SW Longitude']
        self.ne_lat = coordinates['NE Latitude']
        self.ne_lon = coordinates['NE Longitude']
        #print(lat, lon, self.sw_lat, self.sw_lon, self.ne_lat, self.ne_lon, sep='\n')
        self.field_x = (lon-self.sw_lon)*(self.img.shape[1]/(self.ne_lon-self.sw_lon))
        self.field_x = int(round(self.field_x))
        self.field_y = (self.ne_lat-lat)*(self.img.shape[0]/(self.ne_lat-self.sw_lat))
        self.field_y = int(round(self.field_y))

    def get_geojson_polygon(self):
        def invert_pixels(img):
            img = cv2.bitwise_not(img)
            return img

        def get_circle_mask(center, radius, img):
            mask = np.zeros(img.shape, dtype=np.uint8)
            cv2.circle(mask, (center[0],center[1]), radius, (255,255,255), -1)
            return mask

        def get_circle(center, radius, img):
            mask = get_circle_mask(center, radius, img)
            result = img & mask
            
            mask_sum = np.sum(mask)
            result_sum = np.sum(result)
            coverage = (result_sum-mask_sum)/mask_sum
            return [result,coverage]

        def get_ring(center,radius, img, indx):
            mask = np.zeros(img.shape, dtype=np.uint8)
            cv2.circle(mask, (center[0],center[1]), radius, (255,255,255), 1)
            
            img_ring = ((mask)&(invert_pixels(img))).nonzero()
            img_ring = np.array(img_ring)
            
            if img_ring.size != 0:
                img_ring = np.array([img_ring[1],img_ring[0]])
                angles = np.arctan2(img_ring[1] - center[1], img_ring[0] - center[0])
                angles = np.where(angles >= 0, angles, angles + 2*math.pi)
                distances = np.sqrt((img_ring[0] - center[0])**2 + (img_ring[1] - center[1])**2)
                angle1 = np.arctan2(img_ring[1] - center[1] + 2, img_ring[0] - center[0] - 2)
                angle2 = np.arctan2(img_ring[1] - center[1] - 2, img_ring[0] - center[0] + 2)
                angle3 = np.arctan2(img_ring[1] - center[1] + 2, img_ring[0] - center[0] + 2)
                angle4 = np.arctan2(img_ring[1] - center[1] - 2, img_ring[0] - center[0] - 2)
                angle1 = np.where(angle1 >= 0, angle1, angle1 + 2*math.pi)
                angle2 = np.where(angle2 >= 0, angle2, angle2 + 2*math.pi)
                angle3 = np.where(angle3 >= 0, angle3, angle3 + 2*math.pi)
                angle4 = np.where(angle4 >= 0, angle4, angle4 + 2*math.pi)
                img_ring = np.vstack((img_ring, angles, distances))
                for i in range(len(img_ring[0])):
                    a1 = 0.0
                    a2 = 0.0
                    if angles[i] < math.pi/2 or ((angles[i] < 3*math.pi/2) and (angles[i] >= math.pi)):
                        a1 = angle1[i]
                        a2 = angle2[i]
                    else:
                        a1 = angle3[i]
                        a2 = angle4[i]
                    a_min = min(a1,a2)
                    a_max = max(a1,a2)
                    if a_min < 0.75 and a_max > 5.5:
                        a_min = a_min + 2*math.pi
                    r = int(img.shape[1])            
                    cv2.ellipse(img, (center[0],center[1]), (r,r), 0,
                                np.rad2deg(a_min), np.rad2deg(a_max), 255, -1)

                return [img_ring, img]
            img_ring = np.array([[],[],[],[]])
            return [img_ring, img]

        def get_imgs(img):    
            cA = []
            eA = []
            hcorner_centroids = []
            imgs = []
            #print(img.shape)
            for i in range(4):
                # get harris corner image
                cImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                cImg = np.float32(cImg)
                dst = cv2.cornerHarris(cImg,2,1,0.04)
                dst = cv2.dilate(dst,None)
                cImg[dst>0.01*dst.max()]=[0]
                ret, cImg = cv2.threshold(cImg, 90, 255, cv2.THRESH_BINARY)
                cImg = np.uint8(cImg)
                cA.append(cImg)
                
                # find centroids of harris corner
                ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
                dst = cv2.dilate(dst, None, iterations=1)
                dst = np.uint8(dst)
                ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
                hcorner_centroids.append(centroids)
                
                eImg = cv2.Canny(img,50,150)
                eImg = invert_pixels(eImg)
                ret, eImg = cv2.threshold(eImg, 90, 255, cv2.THRESH_BINARY)
                eA.append(eImg)
                
                imgs.append(img)
                img = cv2.pyrDown(img)
                
            return [cA, hcorner_centroids, eA, imgs]

        def print_imgs(imgs):
            imgs = imgs[3] + imgs[2] + imgs[0]
            plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')

            for i in range(len(imgs)):
                plt.subplot(math.ceil(len(imgs)/2),2,i+1)
                plt.imshow(imgs[i],'gray')
                plt.xticks([]),plt.yticks([])
                
            #plt.show()
            plt.savefig("./pyramid.png")

        def print_polygon(img, pts, center):
            pts = pts.reshape((-1,1,2))
            cv2.polylines(img,[pts],True,(150,150,150))
            cv2.circle(img, (center[0],center[1]), 2, (150,0,0), -1)
            
            plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
            plt.imshow(img,'gray')
            plt.xticks([]),plt.yticks([])
            #plt.show()
            plt.savefig("./field2")
            
        def print_img(img, point):
            cv2.circle(img, point, 1, (150,0,0), -1)
            
            plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
            plt.imshow(img,'gray')
            plt.xticks([]),plt.yticks([])
            #plt.show()
            plt.savefig("./field1")
            
        def print_img_points(image, points, i):
            for point in points:
                point = (int(point[0]), int(point[1]))
                cv2.circle(image, point, 1, (150,0,0), -1)
            
            plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
            plt.imshow(image,'gray')
            plt.xticks([]),plt.yticks([])
            plt.show()
            #plt.savefig("./Lidar/%d"%i, transparent=True)

        def get_start_img(imgs):
            img = np.array([0])
            imgA = []
            for i in range(len(imgs[2])):
                img = imgs[2][i]
                center = [int(round(self.field_x/(2**i))),int(round(self.field_y/(2**i)))]
                radius = int(round(img.shape[0]/8))
                img, coverage = get_circle(center, radius, img)
                if coverage < 0.01:
                    return [i, center, radius]
            return  [len(imgs[2])-1, center, radius]

        def closest_point(pt, pts):
            dist = np.sqrt(np.sum((pts - pt)**2, axis=1))
            return [np.amin(dist),np.argmin(dist)]

        def remove_small_angles(p_df, img):
            p_df = p_df.copy()
            
            p0 = cv2.approxPolyDP(p_df.values, int(round(img.shape[1]*.01)), True).reshape((-1,2))
            p1 = np.roll(p0, 2)
            p2 = np.roll(p0, -2)
            
            v1 = p1 - p0
            v2 = p2 - p0

            v1_u = v1 / np.array([np.linalg.norm(v) for v in v1])[:, np.newaxis]
            v2_u = v2 / np.array([np.linalg.norm(v) for v in v2])[:, np.newaxis]

            angle = np.arccos(np.clip(np.einsum('ij,ij->i',v1_u,v2_u), -1.0, 1.0))
            zeros = np.zeros(p0.shape, dtype=np.uint8)
            p0 = np.where(angle[:, None] < 1.2, p0, zeros)
            p0 = p0[np.nonzero(p0)].reshape((-1,2))
            p0 = p0.T
            row_count = p_df.shape[0]
            mask_x = np.logical_not(p_df['x'].isin(p0[0]))
            mask_y = np.logical_not(p_df['y'].isin(p0[1]))
            p_df = p_df.loc[mask_x | mask_y]
            new_row_count = p_df.shape[0]
            if row_count != new_row_count and new_row_count > 3:
                return remove_small_angles(p_df, img)
            return p_df

        def clean_polygon(p_df, corners, img):
            p_df = p_df.copy()
            p_df.columns = ['x','y','angle_origin','distance_origin']
            
            # Code for using harris corner image instead of centroids
            corners = invert_pixels(corners).nonzero()
            corners = np.dstack((corners[1], corners[0]))[0]
            
            corners = corners[1:]
            corners = np.round(corners/2).astype(int)

            vertices = p_df[['x','y']].values
            closest_index = np.array([closest_point(pt,corners) for pt in vertices])
            corner_dist = closest_index.T[0]
            closest_index = closest_index.T[1].astype(int)
            p_df['closest_corner_x'] = np.array([corners[x][0] for x in closest_index])
            p_df['closest_corner_y'] = np.array([corners[x][1] for x in closest_index])
            p_df['corner_dist'] = corner_dist

            p_df = p_df.sort_values('corner_dist')
            p_df.reset_index(drop=True, inplace=True)
            p_df = p_df.loc[p_df['corner_dist'] <= p_df['corner_dist'][int(p_df.shape[0]*0.75)]]

            p_df = p_df.sort_values('angle_origin')
            p_df = p_df[['closest_corner_x', 'closest_corner_y']].drop_duplicates()
            
            p_df.columns = ['x', 'y']
            p_df = remove_small_angles(p_df, img)
            
            p_df_a = cv2.approxPolyDP(p_df.values, int(round(img.shape[1]*.01)), True)
            p_df = pd.DataFrame(p_df_a.reshape((-1,2))) 
            p_df.columns = ['x', 'y']
            
            p0 = p_df[['x','y']].values
            p1 = np.roll(p0, 2)
            p2 = np.roll(p0, -2)
            
            length1 = np.sqrt((p1.T[0] - p0.T[0])**2 + (p1.T[1] - p0.T[1])**2)
            length2 = np.roll(length1, -1)
            p_df['length1'] = length1
            p_df['length2'] = length2
            return p_df

        def get_lidar_polygon(center, radius, img, img_corners):
            image = img.copy()
            radius = int(radius/2)
            length = int(image.shape[1]/2) - radius
            pixels = []
            
            for i in range(length):
                ring, image = get_ring(center, radius+i+1, image, i)
                pixels.append(ring)
            
            vertices = np.concatenate(pixels,axis=1)
            vertices_df = pd.DataFrame(vertices.T, columns=['x','y','angle_origin','distance_origin'])
            vertices_df = vertices_df.sort_values('angle_origin')
            vertices_df = clean_polygon(vertices_df, img_corners, img)
            vertices = vertices_df[['x','y']].values
            return vertices.astype(int)

        imgs = get_imgs(self.img)
        #print_imgs(imgs)
        index, center, radius = get_start_img(imgs)
        index2 = index
        if index > 0:
            index2 = index - 1
        pic = imgs[3][index].copy()
        polygon = get_lidar_polygon(center, radius, imgs[2][index], imgs[0][index2])
        #print_polygon(pic,polygon,center)

        geo_polygon = self.points_to_geojson(polygon, imgs[2][index].shape)
        geo_polygon.append(geo_polygon[0])
        geo_polygon = {'type': 'Polygon', 'coordinates': [geo_polygon]}
        return geo_polygon

    def point_to_lon_lat(self, point, img_shape):
        lon = -point[0]*(self.sw_lon-self.ne_lon)/img_shape[1] + self.sw_lon
        lat = point[1]*(self.sw_lat-self.ne_lat)/img_shape[0] + self.ne_lat
        #print(point, img_shape, self.ne_lon-self.sw_lon, point[0]*(self.ne_lon-self.sw_lon)/img_shape[1], sep='\n')
        return [lon,lat]

    def points_to_geojson(self, polygon, img_shape):
        lon_lat_polygon = []
        for point in polygon:
            lon_lat = self.point_to_lon_lat(point, img_shape)
            lon_lat_polygon.append(lon_lat)
        return lon_lat_polygon