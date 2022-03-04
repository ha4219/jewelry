#ifndef DELAUNAY_H
#define DELAUNAY_H

#ifdef __cplusplus
extern "C" {
#endif

	typedef double real;

	typedef struct del_point2d_t {
		real	x, y;
		del_point2d_t(real i, real j) : x(i), y(j) {};
	};

	typedef struct {
		/** input points count */
		unsigned int	num_points;

		/** the input points */
		del_point2d_t* points;

		/** number of returned faces */
		unsigned int	num_faces;

		/** the faces are given as a sequence: num verts, verts indices, num verts, verts indices...
		 * the first face is the external face */
		unsigned int* faces;
	} delaunay2d_t;

	/*
	 * build the 2D Delaunay triangulation given a set of points of at least 3 points
	 *
	 * @points: point set given as a sequence of tuple x0, y0, x1, y1, ....
	 * @num_points: number of given point
	 * @preds: the incircle predicate
	 * @faces: the triangles given as a sequence: num verts, verts indices, num verts, verts indices.
	 *	Note that the first face is the external face
	 * @return: the created topology
	 */
	delaunay2d_t* delaunay2d_from(del_point2d_t* points, unsigned int num_points);

	/*
	 * release a delaunay2d object
	 */
	void				delaunay2d_release(delaunay2d_t* del);


	typedef struct {
		/** input points count */
		unsigned int	num_points;

		/** input points */
		del_point2d_t* points;

		/** number of triangles */
		unsigned int	num_triangles;

		/** the triangles indices v0,v1,v2, v0,v1,v2 .... */
		unsigned int* tris;
	} tri_delaunay2d_t;

	/**
	 * build a tri_delaunay2d_t out of a delaunay2d_t object
	 */
	tri_delaunay2d_t* tri_delaunay2d_from(delaunay2d_t* del);

	/**
	 * release a tri_delaunay2d_t object
	 */
	void				tri_delaunay2d_release(tri_delaunay2d_t* tdel);

#ifdef __cplusplus
}
#endif

#endif // DELAUNAY_H
