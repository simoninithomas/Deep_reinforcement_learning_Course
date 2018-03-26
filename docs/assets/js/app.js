/************************************************************************************************************************
 ************************************************************************************************************************
 ************************************************************************************************************************
 *
 *                                                         app.js
 * @Author : SIMONINI Thomas, 2016 simonini_thomas@outlook.fr
 ************************************************************************************************************************
 ************************************************************************************************************************
 ************************************************************************************************************************/
angular.module('app', ['ngRoute', 'ngAnimate'])

    .config(function ($routeProvider) {
            $routeProvider
                .when('/', {
                    templateUrl: 'templates/home.html'
                })
                .when('/terms-of-use', {
                    templateUrl: 'templates/termsfeed-terms-service-html-english.html'
                })



                .otherwise('/')
        }
    )

    .run(['$rootScope', function ($rootScope) {
        //create a new instance
        new WOW().init();

        $rootScope.$on('$routeChangeStart', function (next, current) {
            //when the view changes sync wow
            new WOW().sync();
        });
    }])

    
